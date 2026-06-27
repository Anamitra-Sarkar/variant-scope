FROM node:20-bookworm-slim

WORKDIR /app

COPY package.json ./
RUN npm install --no-fund --no-audit

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
