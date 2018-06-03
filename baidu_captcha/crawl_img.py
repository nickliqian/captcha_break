import requests

url = "https://xin.baidu.com/check/getCapImg?t=1527915770509"
headers = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36",
}

for i in range(1, 101):
    print(i)
    response = requests.get(url=url, headers=headers)
    with open("./img/{}.png".format(i), "wb") as f:
        f.write(response.content)