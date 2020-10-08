# 開発環境整備
1. conda create -n FaceOmni python=3.6
2. pip install -r requirements.txt

## ローカル環境の実行手順
1. `cd Client/static`  
2. `npm run dev`  
3. Open new terminal and go to dir <src>/Server: `cd Server`  
4. `python main.py` 

## DigitalOcean サーバーでの実行手順
1. Login via ssh
2. Change to root: `sudo -i`
3. Install python3's libraries:  
`pip install -r requirements_ubuntu16.txt`
5. Install Nodejs:  
`curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh`  
`sudo bash nodesource_setup.sh`
6. Install pm2  
`npm install -g pm2`
7. Install others libraries:  
`sudo apt install libsm6 libxrender1 libxext-dev` 
8. Change dir to <source code>/Server/
9. Start server by pm2: `pm2 start startapp.sh`
