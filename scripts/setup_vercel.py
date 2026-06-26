#!/usr/bin/env python3
"""
Vercel Project Setup Script for VariantScope.
Run this AFTER creating the project on Vercel dashboard.

Steps to create Vercel project:
1. Go to https://vercel.com
2. Click "Add New" → "Project"
3. Import "Anamitra-Sarkar/variant-scope"
4. Set Root Directory to "frontend"
5. Framework preset: Next.js
6. Build Command: next build
7. Output Directory: .next
8. Click "Deploy"

Then run this script to set environment variables and trigger a production deploy.
"""

import os
import sys
import json
import urllib.request
import urllib.error

VERCEL_TOKEN = os.environ.get("VERCEL_TOKEN", "")
TEAM_ID = "team_gXozpOHE3KrJsdcyHVaspIRG"
PROJECT_NAME = "variantscope"
GITHUB_REPO_ID = 1281097358

env_vars = {
    "NEXT_PUBLIC_API_URL": "https://variantscope-api.onrender.com",
}


def api_call(method, path, data=None):
    url = f"https://api.vercel.com{path}"
    if TEAM_ID:
        url += f"{'&' if '?' in path else '?'}teamId={TEAM_ID}"

    headers = {
        "Authorization": f"Bearer {VERCEL_TOKEN}",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data else None,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"  HTTP {e.code}: {body[:200]}")
        return None


def main():
    # Step 1: Add environment variables
    print("Step 1: Setting environment variables...")
    for key, value in env_vars.items():
        result = api_call("POST", f"/v9/projects/{PROJECT_NAME}/env", {
            "key": key,
            "value": value,
            "type": "plain",
            "target": ["production", "preview", "development"],
        })
        if result:
            print(f"  ✅ {key} set")
        else:
            print(f"  ❌ {key} failed")

    # Step 2: Trigger production deployment
    print("\nStep 2: Triggering production deployment...")
    result = api_call("POST", "/v13/deployments", {
        "name": PROJECT_NAME,
        "gitSource": {
            "type": "github",
            "repoId": GITHUB_REPO_ID,
            "ref": "main",
        },
        "projectSettings": {
            "rootDirectory": "frontend",
            "framework": "nextjs",
            "buildCommand": "next build",
            "outputDirectory": ".next",
            "installCommand": "npm install",
        },
    })
    if result and "url" in result:
        print(f"  ✅ Deployment triggered! URL: https://{result['url']}")
    else:
        print(f"  ❌ Deployment failed")


if __name__ == "__main__":
    main()
