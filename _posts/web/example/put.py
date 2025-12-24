import requests

url = 'https://httpbin.org/put'
data = {
    'id': '123',
    'name': 'Boon Hong'
}

response = requests.put(url, data=data)
print(response.status_code)
print(response.text)