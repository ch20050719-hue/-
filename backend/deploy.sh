#!/bin/bash
set -e

APP_DIR="/opt/mask-detection"
APP_PORT=5000

apt update -qq
apt install -y -qq python3 python3-pip python3-venv
mkdir -p $APP_DIR
cp -r backend $APP_DIR/
cp -r yolov5 $APP_DIR/
cd $APP_DIR
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt -q
pip install gunicorn -q
SERVICE_NAME="mask-detection"
cat > /etc/systemd/system/${SERVICE_NAME}.service << 'SYSEOF'
[Unit]
Description=Mask Detection API
After=network.target
[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}/backend
ExecStart=${APP_DIR}/venv/bin/gunicorn -w 1 -b 0.0.0.0:${APP_PORT} app:app
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
SYSEOF
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl restart $SERVICE_NAME
ufw allow $APP_PORT/tcp 2>/dev/null || true
IP=$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_VPS_IP')
echo "=== Done ==="
echo "API: http://${IP}:${APP_PORT}"
