#!/usr/bin/env python3
import requests

if __name__ == "__main__":

    ip = requests.get("https://checkip.amazonaws.com").text.strip()
    print(ip)
