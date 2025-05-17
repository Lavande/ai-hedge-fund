from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import os
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage
from typing import Dict, Any
from src.llm.models import get_model, ModelProvider

router = APIRouter(prefix="/crypto-analysis")

class CryptoAnalysisRequest(BaseModel):
    """Request model for crypto analysis"""
    data: Dict[str, Any]

@router.post("/localize", response_model=Dict[str, str])
async def localize_crypto_analysis(request: CryptoAnalysisRequest = Body(...)):
    """
    Convert crypto analysis to localized markdown for Chinese investors
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="DeepSeek API key not found. Please make sure DEEPSEEK_API_KEY is set in your environment variables."
            )
        
        # Get model using the utility function from the codebase
        model = get_model("deepseek-reasoner", ModelProvider.DEEPSEEK)
        
        # Construct the prompt with user data
        prompt = f"""
正在对加密货币的基本面分析结果的C端用户展示设计，目标用户是中国个人投资者（可能不具备专业投资知识）
1. 需要针对里面已有的参数进行本土化，每个参数对应的中文名称
2. 把结果输出为markdown呈现
3. 本轮对话的结果只有markdown

以下是需要进行本土化和转换为markdown的加密货币分析数据:

{request.data}
"""
        
        # Call the model
        response = model.invoke([HumanMessage(content=prompt)])
        
        # Return the markdown response
        return {"markdown": response.content}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") 