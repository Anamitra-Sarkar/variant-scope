FROM node:20-bookworm-slim

WORKDIR /app

COPY package.json ./
RUN npm install 2>&1

COPY . .

EXPOSE 3000

CMD npm run build 2>&1 && npm start -p ${PORT:-3000}
