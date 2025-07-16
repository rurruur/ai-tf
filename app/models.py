"""
도메인 모델 정의
공통으로 사용되는 데이터 구조들을 정의합니다.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated


class ProductItem(BaseModel):
    """개별 상품 정보"""
    erp_code: Annotated[str, Field(description="상품 ERP 코드", example="DXSH7244N-ZBS")]
    product_name: Annotated[str, Field(description="상품명", example="웨이브 샌디")]
    price: Annotated[int, Field(ge=0, description="가격 (원)", example=109000)]
    color_code: Annotated[str, Field(description="색상 코드", example="ZBS")]
    color_name: Annotated[str, Field(description="색상명", example="Black")]
    category: Annotated[str, Field(description="카테고리", example="SHOES,SHOES,라이프,스니커즈,슬리퍼/샌들,슬리퍼/샌들,웨이브")]
    gender: Annotated[str, Field(description="성별 (남성/여성/공용)", example="공용")]
    status: Annotated[str, Field(description="상품 상태", example="판매중")]
    image_url: Annotated[str, Field(description="이미지 URL", example="https://static-resource.discovery-expedition.com/cdn-cgi/image/format=auto,fit=contain,onerror=redirect/images/goods/ec/X24NDXSH7244NZBS/thnail/3A91220ED23C4E47B9DA888A6CEE88C6.png")]
    detail_url: Annotated[str, Field(description="상세 페이지 URL", example="https://www.discovery-expedition.com/product-detail/DXSH7244N-ZBS")]
    size: Annotated[str, Field(description="사이즈 정보", example="[]")]
    description: Annotated[str, Field(description="상품 설명", example="웨이브 샌디 관련 키워드: 고윤정,고윤정샌달,고윤정샌들,고윤정신발,고윤정착용,고윤정착장,남성아웃도어,남성운동화,바닷가,비오는날,샌달운동화,샌동화,샌들,샌들운동화,스니커샌달,스니커샌들,슬리퍼,신발,여름신발,여름운동화,여성아웃도어,여성운동화,여행,웨이브샌디,접지력,캠핑,퀵레이스")]
    relevance_score: Annotated[float, Field(default=0.0, ge=0.0, le=1.0, description="관련도 점수", example=0.35)]


class SearchResponse(BaseModel):
    """검색 응답 형태"""
    success: Annotated[bool, Field(default=True, description="요청 성공 여부", example=True)]
    message: Annotated[str, Field(default="", description="응답 메시지", example="총 15개의 상품을 찾았습니다. 상위 5개 상품을 추천드립니다.")]
    query_analysis: Annotated[Dict[str, Any], Field(default_factory=dict, description="쿼리 분석 결과", example={
        "original_query": "여름 샌들 추천해줘",
        "extracted_keywords": ["여름 샌들"],
        "detected_color": ""
    })]
    recommended_products: Annotated[List[ProductItem], Field(default_factory=list, description="추천 상품 목록")]
    total_count: Annotated[int, Field(default=0, ge=0, description="총 상품 수", example=15)]


class QueryRequest(BaseModel):
    """검색 요청 형태"""
    query: Annotated[str, Field(min_length=1, description="검색 쿼리", example="여름 샌들 추천해줘")]


# API 응답 타입 (SearchResponse와 동일하지만 명시적으로 분리)
QueryResponse = SearchResponse