# Deploy to Render — Step by Step

## Prerequisites
- GitHub account (free)
- Render account (free tier available at render.com)
- Git installed on your machine

## Step 1: Initialize Git & Push to GitHub

### 1a. Initialize Git repo locally
```powershell
cd D:\Camera_Project
git init
git add .
git commit -m "Initial commit: Streamlit emotion classifier app"
```

### 1b. Create a GitHub repository
1. Go to [github.com/new](https://github.com/new)
2. **Repository name**: `emotion-classifier` (or any name)
3. **Description**: `Real-time emotion detection with Streamlit`
4. Choose **Public** (required for Render free tier to access)
5. Click **Create repository**

### 1c. Push to GitHub (copy commands from GitHub after repo creation)
```powershell
git remote add origin https://github.com/<YOUR_USERNAME>/emotion-classifier.git
git branch -M main
git push -u origin main
```

Replace `<YOUR_USERNAME>` with your actual GitHub username.

## Step 2: Deploy to Render

### 2a. Sign up for Render
1. Go to [render.com](https://render.com)
2. Click **Sign up** → choose **GitHub**
3. Authorize Render to access your GitHub account
4. Complete signup

### 2b. Create a Web Service
1. In Render dashboard, click **New +** (top right)
2. Select **Web Service**
3. Click **Connect a repository** → find and select `emotion-classifier`
4. Click **Connect**

### 2c. Configure the Service
Fill in these fields:

| Field | Value |
|-------|-------|
| **Name** | `emotion-classifier` |
| **Environment** | `Docker` |
| **Branch** | `main` |
| **Build Command** | *(leave empty — Dockerfile will handle it)* |
| **Start Command** | *(leave empty — Dockerfile will handle it)* |
| **Plan** | `Free` (or upgrade if desired) |

### 2d. Advanced Options (Optional but Recommended)
- **Instance Type**: Free (starts with limited resources)
- **Auto-deploy**: Enable (auto-deploy on git push)
- **Health Check Path**: Leave default (`/`)

### 2e. Deploy
Click **Create Web Service** → Render will build and deploy automatically.

**Build time**: ~5–10 minutes (first time).

## Step 3: Access Your App

Once deployed, Render will give you a URL like:
```
https://emotion-classifier-<random>.onrender.com
```

Open that URL in your browser → app is live!

## Troubleshooting

### Build Fails
- Check build logs in Render dashboard (Logs tab).
- Common issue: missing system libs. Dockerfile should handle MediaPipe deps.
- If error mentions `libgl1` or `ffmpeg`, ensure Dockerfile includes them (it does).

### App crashes after deploy
- Check Logs tab for errors.
- Likely cause: memory. Upgrade instance type if needed.

### Webcam doesn't work
- Browser security: some browsers require HTTPS for webcam access (Render uses HTTPS).
- If blocked, use **Upload Image** tab instead.

## Auto-Deploy (Optional)
If you enable auto-deploy in Render settings:
- Every git push to `main` triggers a new build.
- Render rebuilds and deploys automatically.

```powershell
git add .
git commit -m "Update app"
git push origin main
# Render automatically builds and deploys
```

## Tips
- **Free tier limits**: Render free tier has memory/CPU limits. If you see slowness, upgrade to a paid plan.
- **Persistent state**: Streamlit Cloud or Docker-based deployments are stateless; session data resets on redeploy.
- **Custom domain**: Render supports custom domains (Settings → Custom Domain).

## Done!
Your Streamlit emotion classifier is now live on the internet. Share the URL with others!
