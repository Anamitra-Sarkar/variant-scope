FROM node:20-bookworm-slim AS builder

WORKDIR /app

COPY package.json ./
RUN npm install --no-fund --no-audit 2>&1

COPY . .
RUN npm run build 2>&1

EXPOSE 3000

CMD ["npm", "start"]
