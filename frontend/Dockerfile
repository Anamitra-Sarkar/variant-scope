FROM node:20-bookworm-slim

WORKDIR /app

COPY package.json ./
RUN npm install 2>&1

COPY . .

EXPOSE 3000

RUN npm run build 2>&1

CMD node_modules/.bin/next start -p ${PORT:-3000}
