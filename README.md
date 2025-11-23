# lambda-monitoring-api

You can get the server running a subset of the following commands.

```bash
cd lilbro
# git clone https://github.com/LittleBrotherAI/lambda-monitoring-api.git
cd lambda-monitoring-api
# python3 -m venv .venv
source .venv/bin/activate
# pip install fastapi uvicorn
# pip freeze > requirements.txt
pip install -r requirements.txt
sudo su
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 80 2>&1 | tee server.logs
```

Then the server will be running! You can test the connection with `curl http://<ip-address>:80/` and should get a welcome message :)
You can also ssh into the server and check the file `server.logs` to see what's going wrong.

TODO:
- tarik add the remaining callbackurls
- change it so that surprisal.py does not redownload models every time at start and when called
- add cot_untrusted_trusted_similarity and surprisal to main
