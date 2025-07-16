from typing import Any, TypedDict, List, Optional
import difflib
import json
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma

from app.models import ProductItem, SearchResponse

def convert_to_relevance_score(similarity_score: float) -> float:
    """
    벡터 유사도 점수를 0-1 범위의 relevance score로 변환
    유사도 점수가 낮을수록 더 유사함 (거리 기반)
    """
    if similarity_score <= 0:
        return 1.0
    elif similarity_score >= 2.0:
        return 0.0
    else:
        return round(1.0 - (similarity_score / 2.0), 3)

class GraphState(TypedDict, total=False):
    query: str
    messages: List[Any]
    context: List[Any]
    answer: str
    chat_history: List[Any]
    extracted_color: Optional[dict]
    keywords: str
    json_response: dict

class LangchainService:
  
  def __init__(self, config: dict, system_prompt: Optional[str] = None):
    self.config = config
    self.system_prompt = system_prompt or "당신은 도움이 되는 제품 검색 어시스턴트입니다."
    self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    self.vector_store = Chroma(
      collection_name="dxm_products",
      embedding_function=self.embeddings,
      persist_directory="./chroma_db",
    )
    self.graph = self._create_graph()
    self.color_data = self._load_color_data()
  
  def _create_graph(self):
    workflow = StateGraph(GraphState)
    
    workflow.add_node("analyze_query", self._analyze_query)
    workflow.add_node("analyze_color", self._analyze_color)
    workflow.add_node("combine_analysis", self._combine_analysis)
    workflow.add_node("retrieve_context", self._retrieve_context)
    workflow.add_node("generate_answer", self._generate_answer)
    
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge(START, "analyze_color")
    
    workflow.add_edge("analyze_query", "combine_analysis")
    workflow.add_edge("analyze_color", "combine_analysis")
    
    workflow.add_edge("combine_analysis", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # print(workflow)
    
    return workflow.compile()
  
  def _analyze_query(self, state: GraphState) -> GraphState:
    query = state["query"]
    
    # 키워드 추출 로직
    prompt = f"""
    사용자 질문에서 상품 검색에 필요한 핵심 키워드만 추출해주세요.
    상품명, 브랜드, 특징, 용도, 스타일 등을 포함하세요.
    
    질문: {query}
    
    키워드만 간단히 답변해주세요 (예: "패딩 따뜻한 겨울")
    """
    
    try:
        response = self.llm.invoke(prompt)
        keywords = str(response.content).strip()
        return {"keywords": keywords, "messages": [HumanMessage(content=query)]}
    except Exception as e:
        print(f"키워드 추출 실패: {e}")
        return {"keywords": query, "messages": [HumanMessage(content=query)]}
  
  def _load_color_data(self):
    """dxm_colors.json 파일에서 색상 정보 로드"""
    try:
        with open('./dxm_colors.json', 'r', encoding='utf-8') as f:
            color_data = json.load(f)
        print("✅ 색상 데이터 로드 완료")
        return color_data
    except FileNotFoundError:
        print("❌ './dxm_colors.json' 파일을 찾을 수 없습니다.")
        return []
    except Exception as e:
        print(f"❌ 색상 데이터 로드 중 오류: {e}")
        return []
  
  def _analyze_color(self, state: GraphState) -> GraphState:
    query = state["query"]
    
    # JSON 파일에서 색상 데이터 로드
    color_data = self.color_data
    if not color_data:
        return {"extracted_color": None}
    
    # 색상 그룹명 목록 생성
    available_colors = [color["COLOR_GRP_NM"] for color in color_data]
    
    color_prompt = f"""
    다음 텍스트에서 색상명을 추출해주세요. 
    추출할 수 있는 색상이 다음 목록에 있는지 확인하고, 가장 적절한 색상 하나만 영어로 답변해주세요.
    
    가능한 색상 목록:
    {', '.join(available_colors)}
    
    규칙:
    1. 위 목록에 있는 색상 중에서만 선택
    2. 가장 적절한 색상 하나만 영어 대문자로 답변
    3. 해당하는 색상이 없으면 "NONE"이라고 답변
    4. 설명 없이 색상명만 답변
    
    텍스트: {query}
    """
    
    try:
        response = self.llm.invoke(color_prompt)
        extracted_color = str(response.content).strip().upper()
        
        # AI 응답이 유효한 색상인지 확인
        if extracted_color in available_colors:
            # 해당 색상의 상세 정보 찾기
            color_info = next((color for color in color_data if color["COLOR_GRP_NM"] == extracted_color), None)
            return {"extracted_color": color_info}
        elif extracted_color == "NONE":
            return {"extracted_color": None}
        else:
            # AI가 유효하지 않은 응답을 한 경우, 문자열 매칭으로 대체
            user_input = query.upper().strip()
            closest_matches = difflib.get_close_matches(
                user_input, 
                available_colors, 
                n=1, 
                cutoff=0.6
            )
            if closest_matches:
                color_info = next((color for color in color_data if color["COLOR_GRP_NM"] == closest_matches[0]), None)
                return {"extracted_color": color_info}
            else:
                return {"extracted_color": None}
            
    except Exception as e:
        print(f"색상 추출 중 오류 발생: {e}")
        return {"extracted_color": None}
  
  def _retrieve_context(self, state: GraphState) -> GraphState:
    keywords = state.get("keywords", state["query"])
    extracted_color = state.get("extracted_color")
    
    search_query = keywords
    
    # 색상 필터가 있으면 메타데이터 필터 적용
    filter_dict = None
    if extracted_color and extracted_color.get("COLOR_CD_LIST"):
        # COLOR_CD_LIST에서 개별 컬러 코드들을 추출
        color_codes = [code.strip() for code in extracted_color["COLOR_CD_LIST"].split(",")]
        
        # $in 연산자를 사용해서 COLOR_CD_LIST의 모든 컬러 코드와 매칭
        filter_dict = {
            "$or": [
                {"color_code": {"$in": color_codes}},
                {"color_group": extracted_color.get("COLOR_GRP_NM", "")},
                {"color_group_code": extracted_color.get("COLOR_GRP_CD", "")}
            ]
        }
        print(f"Color filter applied: {extracted_color.get('COLOR_GRP_NM', '')} with codes: {color_codes[:3]}...")
    
    # 벡터 검색 수행
    k = 20
    try:
        if filter_dict:
            docs = self.vector_store.similarity_search_with_score(
                search_query, 
                k, 
                filter=filter_dict
            )
        else:
            docs = self.vector_store.similarity_search_with_score(search_query, k)
    except Exception as e:
        print(f"Filter search failed, falling back to basic search: {e}")
        # 필터 검색이 실패하면 기본 검색으로 폴백
        docs = self.vector_store.similarity_search_with_score(search_query, k)
    
    print(f"Retrieved {len(docs)} documents for keywords: {keywords}")
    if extracted_color:
        print(f"With color filter: {extracted_color.get('COLOR_GRP_NM', 'None')}")
    
    for i, [doc, score] in enumerate(docs[:10]):
        print(f"Doc {i} Score: {score:.4f}")
        print(f"Doc {i}: {doc.page_content[:100]}...")
        if hasattr(doc, 'metadata'):
            print(f"Metadata: {doc.metadata}")
    
    # docs와 score를 함께 전달
    return {"context": docs}
  
  def _generate_answer(self, state: GraphState) -> GraphState:
    query = state["query"]
    context_docs_with_scores = state["context"]  # (doc, score) 튜플 리스트
    
    unique_products = {}
    product_items = []
    
    for doc, score in context_docs_with_scores:
        erp_code = doc.metadata.get('erp_code', '')
        if erp_code and erp_code not in unique_products:
            unique_products[erp_code] = doc
            
            product_item = ProductItem(
                erp_code=erp_code,
                product_name=doc.metadata.get('product_name', ''),
                price=doc.metadata.get('price', 0),
                color_code=doc.metadata.get('color_code', ''),
                color_name=doc.metadata.get('color', ''),
                category=doc.metadata.get('category', ''),
                gender=doc.metadata.get('gender', ''),
                status=doc.metadata.get('status', ''),
                image_url=doc.metadata.get('image_url', ''),
                detail_url=doc.metadata.get('detail_url', ''),
                size=doc.metadata.get('size', ''),
                description=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                relevance_score=convert_to_relevance_score(score)
            )
            product_items.append(product_item)
    
    # 상위 5개 상품만 선택
    top_products = product_items[:5]
    
    query_analysis = {
        "original_query": query,
        "extracted_keywords": state.get("keywords", ""),
        "detected_color": state.get("extracted_color", {}).get("COLOR_GRP_NM", "") if state.get("extracted_color") else ""
    }
    
    if top_products:
        ai_message = f"총 {len(product_items)}개의 상품을 찾았습니다. 상위 {len(top_products)}개 상품을 추천드립니다."
    else:
        ai_message = "죄송합니다. 검색 조건에 맞는 상품을 찾을 수 없습니다. 다른 키워드로 다시 검색해보세요."
    
    try:
        search_response = SearchResponse(
            success=True,
            message=ai_message,
            query_analysis=query_analysis,
            recommended_products=top_products,
            total_count=len(product_items)
        )
        
        json_response = search_response.model_dump()
        
        return {
            "answer": ai_message,
            "json_response": json_response
        }
    except Exception as e:
        print(f"❌ JSON 응답 생성 중 오류: {e}")
        error_response = SearchResponse(
            success=False,
            message="응답 생성 중 오류가 발생했습니다.",
            query_analysis=query_analysis,
            recommended_products=[],
            total_count=0
        )
        return {
            "answer": "죄송합니다. 응답 생성 중 오류가 발생했습니다.",
            "json_response": error_response.model_dump()
        }
  
  def _combine_analysis(self, state: GraphState) -> GraphState:
    """분석 결과를 결합하는 노드"""
    
    print(f"Keywords: {state.get('keywords', '')}")
    print(f"Extracted Color: {state.get('extracted_color', None)}")
    
    return {}
  
  def call(self, query: str) -> dict:
    initial_state: GraphState = {
        "query": query,
        "messages": [],
        "context": [],
        "answer": "",
        "chat_history": [],
        "extracted_color": None,
        "keywords": "",
        "json_response": {}
    }
    
    result = self.graph.invoke(initial_state)

    return result.get("json_response", {
        "success": False,
        "message": "응답 생성에 실패했습니다.",
        "query_analysis": {},
        "recommended_products": [],
        "total_count": 0
    })
  