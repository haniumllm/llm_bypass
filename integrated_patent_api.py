# ====================================
# 특허 검색, 유사도 계산, 우회전략 통합 API
# pip install fastapi uvicorn requests xmltodict openai python-dotenv numpy
# ====================================

import os
import json
import xmltodict
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ====================================
# 1) 환경 설정
# ====================================
load_dotenv()

KIPRIS_API_KEY = os.getenv("KIPRIS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not KIPRIS_API_KEY:
    raise RuntimeError("KIPRIS_API_KEY 환경변수가 설정되지 않았습니다.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
BAD_ABS = {"", None, "내용 없음", "내용 없음."}

# ====================================
# 2) FastAPI 앱 설정
# ====================================
app = FastAPI(title="Integrated Patent Analysis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================
# 3) 데이터 모델
# ====================================
class PatentAnalysisRequest(BaseModel):
    user_patent_description: str  # 사용자의 특허 아이디어/설명
    search_keyword: Optional[str] = None  # KIPRIS 검색용 키워드 (선택적)
    max_results: Optional[int] = 10

class PatentInsight(BaseModel):
    application_number: str
    title: str
    abstract: str
    ipc_number: Optional[str] = None
    register_status: Optional[str] = None
    detail_url: Optional[str] = None
    similarity_score: str            # "000" ~ "100"
    cosine_similarity: float         # 실제 코사인 값
    strategies: List[str]            # 2~3문장 전략 리스트
    technical_alternatives: List[str]
    risk_assessment: str             # HIGH/MEDIUM/LOW

class IntegratedResponse(BaseModel):
    user_query: str
    search_keyword: str
    total_count: int
    analysis_summary: str
    items: List[PatentInsight]

# ====================================
# 4) 유틸리티 함수들
# ====================================
def make_text(title: str, abstract: str) -> str:
    """제목과 초록을 결합하여 임베딩용 텍스트 생성"""
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    return title if abstract in BAD_ABS else f"{title} {abstract}"

def embed_texts(texts: List[str]) -> np.ndarray:
    """텍스트 리스트를 임베딩 벡터로 변환"""
    try:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
        arr = [np.array(d.embedding, dtype=float) for d in resp.data]
        return np.vstack(arr)  # (N, D)
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 실패: {e}")

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 행렬 간의 코사인 유사도 계산"""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T

def extract_search_keywords(text: str) -> str:
    """복잡한 텍스트에서 검색에 적합한 키워드 추출"""
    # 기술 관련 키워드들을 우선 추출
    tech_keywords = []
    
    # 영문 약어나 기술명 추출 (대문자 연속, 하이픈 포함)
    import re
    tech_terms = re.findall(r'[A-Z][A-Z0-9-]*[A-Z0-9]|[A-Z][a-z]+(?:[A-Z][a-z]+)*', text)
    tech_keywords.extend(tech_terms[:3])  # 상위 3개만
    
    # 한글 기술 용어 추출 (객체, 인식, 최적화 등)
    korean_tech = re.findall(r'[가-힣]+(?:객체|인식|최적화|학습|모델|시스템|기술|방법|장치)', text)
    tech_keywords.extend(korean_tech[:2])  # 상위 2개만
    
    if tech_keywords:
        result = ' '.join(tech_keywords)
        print(f"🎯 추출된 검색 키워드: {result}")
        return result
    
    # 키워드 추출 실패 시 원본 텍스트의 앞부분 사용
    words = text.split()[:5]  # 처음 5단어만
    result = ' '.join(words)
    print(f"🔤 기본 검색 키워드: {result}")
    return result

def score3_from_cos(cos_val: float) -> str:
    """코사인 값을 3자리 점수로 변환 (-1~1 -> 000~100)"""
    sim01 = (cos_val + 1.0) / 2.0
    v = int(round(max(0.0, min(1.0, sim01)) * 100))
    return f"{v:03d}"
    """코사인 값을 3자리 점수로 변환 (-1~1 -> 000~100)"""
    sim01 = (cos_val + 1.0) / 2.0
    v = int(round(max(0.0, min(1.0, sim01)) * 100))
    return f"{v:03d}"

# ====================================
# 5) KIPRIS API 함수
# ====================================
def kipris_search(word: str, rows: int = 10, page: int = 1) -> List[Dict[str, Any]]:
    """KIPRIS API를 사용한 특허 검색"""
    
    search_url = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getWordSearch"
    
    url = (
        f"{search_url}?word={quote_plus(word)}&year=0"
        f"&patent=Y&utility=Y"
        f"&numOfRows={rows}&pageNo={page}&ServiceKey={KIPRIS_API_KEY}"
    )
    
    try:
        print(f"🌐 KIPRIS API 호출: {url}")
        r = requests.get(url, timeout=(10, 60))
        r.raise_for_status()
        
        print(f"📡 응답 상태: {r.status_code}")
        print(f"📄 응답 내용 (처음 500자): {r.text[:500]}")
        
        data = xmltodict.parse(r.content)
        print(f"🔍 파싱된 데이터 구조: {list(data.keys()) if data else 'None'}")
        
        # 더 안전한 중첩 딕셔너리 접근
        response = data.get("response") if data else None
        if not response:
            print("❌ 'response' 키가 없습니다.")
            return []
            
        body = response.get("body") if response else None
        if not body:
            print("❌ 'body' 키가 없습니다.")
            return []
            
        items_container = body.get("items") if body else None
        if not items_container:
            print("❌ 'items' 키가 없습니다.")
            return []
            
        items = items_container.get("item") if items_container else None
        if not items:
            print("❌ 'item' 키가 없거나 비어있습니다.")
            return []
        
        result = items if isinstance(items, list) else [items]
        print(f"✅ 검색 결과: {len(result)}건")
        return result
        
    except requests.RequestException as e:
        print(f"KIPRIS API 호출 오류: {e}")
        raise HTTPException(status_code=502, detail=f"KIPRIS API 오류: {e}")
    except Exception as e:
        print(f"데이터 파싱 오류: {e}")
        print(f"원본 응답: {r.text if 'r' in locals() else 'No response'}")
        raise HTTPException(status_code=502, detail=f"데이터 파싱 오류: {e}")

# ====================================
# 6) 유사도 계산 및 결과 생성
# ====================================
def calculate_similarity_and_rank(user_description: str, patent_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """사용자 설명과 특허들 간의 유사도 계산 및 순위화"""
    
    if not patent_items:
        return []
    
    # 문서 텍스트 준비
    docs = []
    for item in patent_items:
        title = item.get("inventionTitle") or item.get("inventionName") or ""
        abstract = item.get("astrtCont") or ""
        combined_text = make_text(title, abstract)
        docs.append((item, title, abstract, combined_text))
    
    # 임베딩 생성: 사용자 설명 + 모든 특허 문서
    all_texts = [user_description] + [doc[3] for doc in docs]
    all_vecs = embed_texts(all_texts)
    
    # 유사도 계산
    user_vec = all_vecs[0:1, :]  # 사용자 벡터
    doc_vecs = all_vecs[1:, :]   # 문서 벡터들
    cos_scores = cosine_matrix(user_vec, doc_vecs).ravel()
    
    # 결과 생성 및 정렬
    results = []
    for (item, title, abstract, _), cos_val in zip(docs, cos_scores):
        app_number = item.get("applicationNumber", "")
        
        result_item = {
            "item": item,
            "title": title,
            "abstract": abstract,
            "application_number": app_number,
            "ipc_number": item.get("ipcNumber", ""),
            "register_status": item.get("registerStatus", ""),
            "detail_url": f"https://kpat.kipris.or.kr/kpat/biblioa.do?method=biblio&applicationNumber={app_number}" if app_number else None,
            "similarity_score": score3_from_cos(cos_val),
            "cosine_similarity": float(cos_val)
        }
        results.append(result_item)
    
    # 유사도 순으로 정렬
    results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return results

# ====================================
# 7) GPT 우회전략 생성
# ====================================
def calculate_risk_assessment(similarity_score: str, register_status: str) -> str:
    """유사도와 등록상태를 기반으로 리스크 평가"""
    score_int = int(similarity_score)
    
    # 등록 상태에 따른 기본 리스크
    status_risk = {
        "등록": 2,      # 등록된 특허 - 높은 리스크
        "공개": 1,      # 공개된 특허 - 중간 리스크  
        "출원": 1,      # 출원 특허 - 중간 리스크
        "포기": 0,      # 포기된 특허 - 낮은 리스크
        "거절": 0,      # 거절된 특허 - 낮은 리스크
    }.get(register_status, 1)  # 기본값은 중간 리스크
    
    # 유사도에 따른 리스크 점수 (0-2)
    if score_int >= 80:
        similarity_risk = 2
    elif score_int >= 60:
        similarity_risk = 1
    else:
        similarity_risk = 0
    
    # 총 리스크 점수 (0-4)
    total_risk = status_risk + similarity_risk
    
    if total_risk >= 3:
        return "HIGH"
    elif total_risk >= 2:
        return "MEDIUM"
    else:
        return "LOW"

def generate_bypass_strategy(user_description: str, patent_info: Dict[str, Any]) -> Dict[str, Any]:
    """유사한 특허에 대한 우회전략 생성"""
    
    system_prompt = """
당신은 특허 분석 및 우회전략 전문가입니다. 
사용자의 특허 아이디어와 유사한 기존 특허를 분석하여, 법적 침해를 피하면서도 사용자의 목적을 달성할 수 있는 우회전략을 제안하세요.

응답은 다음 JSON 형식으로 작성해주세요:
{
    "strategies": [
        "구체적인 우회전략 1 (2-3문장으로 설명)",
        "구체적인 우회전략 2 (2-3문장으로 설명)",
        "구체적인 우회전략 3 (2-3문장으로 설명)"
    ],
    "technical_alternatives": [
        "기술적 대안 1",
        "기술적 대안 2", 
        "기술적 대안 3"
    ]
}

주의사항:
- 사용자의 원래 목적을 달성하면서도 기존 특허를 회피하는 방법 제시
- 법적 침해를 유도하지 마세요
- 실현 가능하고 구체적인 대안을 제시하세요
- 유사도가 높을수록 더 신중한 우회전략 필요
"""

    user_prompt = f"""
**사용자의 특허 아이디어:**
{user_description}

**유사한 기존 특허 (유사도: {patent_info['similarity_score']}/100):**
- 제목: {patent_info['title']}
- 출원번호: {patent_info['application_number']}
- 요약: {patent_info['abstract'][:500] if patent_info['abstract'] else '요약 없음'}
- 등록상태: {patent_info['register_status']}

사용자의 아이디어와 이 기존 특허 사이의 유사도가 {patent_info['similarity_score']}/100 입니다.
사용자가 원하는 기능을 구현하면서도 이 특허를 회피할 수 있는 전략을 제안해주세요.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            result = json.loads(content)
            strategies = result.get('strategies', ['전략 생성 실패'])
            technical_alternatives = result.get('technical_alternatives', ['대안 생성 실패'])
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기본값
            strategies = [f"우회전략 분석 결과: {content[:200]}..."]
            technical_alternatives = ["상세 분석 필요"]
        
        # 객관적인 리스크 평가 계산
        risk_assessment = calculate_risk_assessment(
            patent_info['similarity_score'], 
            patent_info.get('register_status', '')
        )
        
        return {
            "strategies": strategies,
            "technical_alternatives": technical_alternatives,
            "risk_assessment": risk_assessment
        }
        
    except Exception as e:
        print(f"우회전략 생성 오류: {e}")
        return {
            "strategies": [f"우회전략 생성 중 오류 발생: {str(e)}"],
            "technical_alternatives": ["분석 불가"],
            "risk_assessment": calculate_risk_assessment(
                patent_info.get('similarity_score', '050'), 
                patent_info.get('register_status', '')
            )
        }

# ====================================
# 8) API 엔드포인트
# ====================================
@app.post("/analyze-patent", response_model=IntegratedResponse)
async def analyze_patent_with_similarity(request: PatentAnalysisRequest):
    try:
        # search_keyword가 없으면 user_patent_description에서 키워드 추출
        if request.search_keyword:
            effective_keyword = request.search_keyword.strip()
        else:
            effective_keyword = extract_search_keywords(request.user_patent_description)
        
        if not effective_keyword:
            raise HTTPException(status_code=400, detail="검색어가 비어 있습니다.")

        print(f"📥 사용자 특허 설명: {request.user_patent_description[:100]}...")
        print(f"🔍 실제 사용 검색 키워드: {effective_keyword}")

        # 1) KIPRIS 검색 - 실패 시 재시도
        patent_items = []
        try:
            patent_items = kipris_search(effective_keyword, rows=request.max_results, page=1)
        except HTTPException as e:
            # 첫 번째 검색 실패 시, 더 간단한 키워드로 재시도
            if not request.search_keyword:  # 자동 추출된 경우만
                simple_keyword = request.user_patent_description.split()[0]  # 첫 번째 단어만
                print(f"🔄 간단한 키워드로 재시도: {simple_keyword}")
                try:
                    patent_items = kipris_search(simple_keyword, rows=request.max_results, page=1)
                    effective_keyword = simple_keyword
                except:
                    raise e  # 재시도도 실패하면 원래 예외 발생
            else:
                raise e
        
        if not patent_items:
            return IntegratedResponse(
                user_query=request.user_patent_description,
                search_keyword=effective_keyword,
                total_count=0,
                analysis_summary="검색 결과가 없습니다.",
                items=[]
            )

        print(f"🔍 검색된 특허 수: {len(patent_items)}")

        # 2) 유사도 계산
        similarity_results = calculate_similarity_and_rank(
            request.user_patent_description,
            patent_items
        )
        print(f"📊 유사도 계산 완료: {len(similarity_results)}건")

        # 3) 각 특허별 우회전략 생성
        items: List[PatentInsight] = []
        for result in similarity_results[:request.max_results]:
            print(f"⚙️ 우회전략 생성 중: {result['title'][:50]}... (유사도: {result['similarity_score']})")
            
            bypass_strategy = generate_bypass_strategy(
                request.user_patent_description,
                result
            )

            items.append(PatentInsight(
                application_number=result["application_number"],
                title=result["title"],
                abstract=result["abstract"],
                ipc_number=result["ipc_number"],
                register_status=result["register_status"],
                detail_url=result["detail_url"],
                similarity_score=result["similarity_score"],
                cosine_similarity=result["cosine_similarity"],
                strategies=bypass_strategy["strategies"],
                technical_alternatives=bypass_strategy["technical_alternatives"],
                risk_assessment=bypass_strategy["risk_assessment"]
            ))

        # 4) 요약
        avg = np.mean([float(x.similarity_score) for x in items]) if items else 0
        high = len([x for x in items if int(x.similarity_score) >= 70])
        summary = (
            f"총 {len(similarity_results)}건을 분석했고, 상위 {len(items)}건에 대해 우회전략을 생성했습니다. "
            f"평균 유사도: {avg:.1f}/100, 고유사도(70+): {high}건."
        )

        return IntegratedResponse(
            user_query=request.user_patent_description,
            search_keyword=effective_keyword,
            total_count=len(similarity_results),
            analysis_summary=summary,
            items=items
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")
    
@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "message": "Integrated Patent Analysis API is running"}

@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "Integrated Patent Analysis API",
        "version": "2.0.0",
        "description": "특허 유사도 분석 및 우회전략 추천 통합 API",
        "endpoints": {
            "POST /analyze-patent": "특허 유사도 분석 및 우회전략 생성",
            "GET /health": "헬스 체크",
            "GET /": "API 정보"
        },
        "features": [
            "KIPRIS API 특허 검색",
            "OpenAI 임베딩 기반 유사도 계산",
            "GPT 기반 우회전략 생성",
            "코사인 유사도 점수화 (000-100)"
        ]
    }

# ====================================
# 9) 서버 실행
# ====================================
if __name__ == "__main__":
    print("🚀 Integrated Patent Analysis API 시작중...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )