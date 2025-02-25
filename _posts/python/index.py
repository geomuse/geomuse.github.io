from bs4 import BeautifulSoup
import requests

def get_weather_alternative():
    url = "https://weather.com/en-SG/weather/tenday/l/Ipoh+Perak+Malaysia?canonicalCityId=09555dde67cf0f997ad7a6e71d77deaafce05d95a9dcaf0609e96c230ca095d4"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200 :
        soup = BeautifulSoup(response.text, "html.parser")

        weather = soup.find(class_="DailyContent--temp--axgOn DailyContent--tempN--DPsDJ").text
        Humidity = soup.find(class_="DetailsTable--value--pWEVz").text
        return f"Weather: {weather}, Humidity : {Humidity}"
    else : 
        print("status_code not equal to 200.")

# 测试爬取天气
print(get_weather_alternative())