FROM node:20-bookworm-slim
WORKDIR /app
COPY package.json ./
RUN npm install 2>&1
COPY . .
RUN npm run build 2>&1
CMD ["node", "-e", "console.log('build ok')"]
