FROM node:20-bookworm-slim
WORKDIR /app
COPY package.json ./
RUN npm install 2>&1 || (echo "npm install failed with code $?"; cat /root/.npm/_logs/*.log 2>/dev/null || true; exit 1)
CMD ["node", "-e", "console.log('npm install ok')"]
