#!/bin/bash

sudo apt update -y
sudo apt upgrade -y


sudo apt install -y python3 python3-pip python3-venv nginx libgl1-mesa-glx libglib2.0-0 build-essential screen

PROJECT_DIR="/var/www/farovon"
if [ ! -d "$PROJECT_DIR" ]; then
    sudo mkdir -p "$PROJECT_DIR"
fi
sudo chown $USER:$USER "$PROJECT_DIR"


cd "$PROJECT_DIR"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install ultralytics flask opencv-python torch

pip install torchvision

mkdir -p uploads logs

deactivate

NGINX_CONF="/etc/nginx/sites-available/yolo_flask_app"
sudo bash -c "cat > $NGINX_CONF" <<EOL
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /static/ {
        alias $PROJECT_DIR/static/;
    }

    location /uploads/ {
        alias $PROJECT_DIR/uploads/;
    }
}
EOL

sudo ln -s /etc/nginx/sites-available/yolo_flask_app /etc/nginx/sites-enabled

NGINX_CONF="/etc/nginx/sites-available/yolo_flask_app_2"
sudo bash -c "cat > $NGINX_CONF" <<EOL
server {
    listen 8080;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /static/ {
        alias $PROJECT_DIR/static/;
    }

    location /uploads/ {
        alias $PROJECT_DIR/uploads/;
    }
}
EOL

sudo ln -s /etc/nginx/sites-available/yolo_flask_app_2 /etc/nginx/sites-enabled

sudo systemctl restart nginx


echo "Installation is complete. The Flask app should now be running and accessible via Nginx."
echo "If you've set up a domain, it should now be accessible at http://your_domain_or_ip."
echo "You can manage the Nginx service with: sudo systemctl restart nginx"

