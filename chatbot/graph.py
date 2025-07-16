from dotenv import load_dotenv
from langchain_tavily import TavilySearch
import os
import requests
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)

search_tool = TavilySearch(max_results=5, topic="general")


class WeatherQuery(BaseModel):
    loc: str = Field(description="The location name of the city")


@tool(args_schema=WeatherQuery)
def get_weather(loc):
    """ 查询即时天气函数
    :param loc: 必要参数，字符串类型，用户查询天气的具体城市名称，\
    注意，中国的城市需要用对应成是的英文名称代替，例如查询北京市额天气，则loc参数需要输入'Beijing'
    :return: OpenWeather API 查询即时天气的结果，具体URL请求地址为: https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析后的JSON格式对象，并用字符串的形式表示，其中包含了全部重要的天气信息
    """
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": loc,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn",
    }

    response = requests.get(url, params=params)

    data = response.json()

    return json.dumps(data)


tools = [search_tool, get_weather]

model = init_chat_model(
    model="deepseek-v3",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
graph = create_react_agent(model=model, tools=tools)
