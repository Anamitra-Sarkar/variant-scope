FROM node:20-bookworm-slim
WORKDIR /app

# Step 1: create minimal app
RUN mkdir -p src/app
RUN echo 'export default function Page() { return <h1>Hello</h1>; }' > src/app/page.tsx
RUN echo 'export default function RootLayout({ children }) { return <html lang="en"><body>{children}</body></html>; }' > src/app/layout.tsx
RUN echo '{"name":"test","private":true,"scripts":{"build":"next build"}}' > package.json
RUN echo '{}' > next.config.js
RUN echo '{"compilerOptions":{"jsx":"preserve","module":"esnext","moduleResolution":"bundler","strict":true,"noEmit":true},"include":["src/**/*"]}' > tsconfig.json

# Step 2: npm install
RUN npm install next@16 react@19 react-dom@19 2>&1

# Step 3: build
RUN ./node_modules/.bin/next build 2>&1

EXPOSE 3000
CMD ["node_modules/.bin/next", "start", "-p", "3000"]
