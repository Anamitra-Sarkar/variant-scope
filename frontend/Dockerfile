FROM node:20-bookworm-slim
WORKDIR /app

# Test minimal Next.js build
RUN mkdir -p src/app && \
    echo 'export default function Page() { return <h1>Hello</h1>; }' > src/app/page.tsx && \
    echo 'export default function RootLayout({ children }) { return <html lang="en"><body>{children}</body></html>; }' > src/app/layout.tsx && \
    echo '{"name":"test","private":true}' > package.json && \
    echo '{}' > next.config.js && \
    echo '{"compilerOptions":{"jsx":"preserve","module":"esnext","moduleResolution":"bundler","strict":true,"noEmit":true},"include":["src/**/*"]}' > tsconfig.json

RUN npm install next@16 react@19 react-dom@19 2>&1

RUN npx next build 2>&1

CMD ["npx", "next", "start", "-p", "3000"]
