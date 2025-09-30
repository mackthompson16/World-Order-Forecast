# GitHub Pages Deployment Guide

This guide explains how to deploy the React documentation app to GitHub Pages.

## Prerequisites

1. GitHub repository set up
2. Node.js and npm installed
3. Git configured

## Deployment Steps

### 1. Update Repository URLs

First, update the URLs in the following files to match your GitHub username:

**docs/package.json:**
```json
"homepage": "https://your-username.github.io/country-standing-forecast"
```

**docs/src/components/Header.tsx:**
```typescript
href="https://github.com/your-username/country-standing-forecast"
href="https://your-username.github.io/country-standing-forecast"
```

**README.md:**
```markdown
🌐 **[Live Demo & Documentation](https://your-username.github.io/country-standing-forecast)**
```

### 2. Build and Deploy

```bash
# Navigate to docs directory
cd docs

# Install dependencies (if not already done)
npm install

# Build the app
npm run build

# Deploy to GitHub Pages
npm run deploy
```

### 3. Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to Settings → Pages
3. Under "Source", select "Deploy from a branch"
4. Select "gh-pages" branch and "/ (root)" folder
5. Click Save

### 4. Access Your Site

Your documentation will be available at:
`https://your-username.github.io/country-standing-forecast`

## Development Workflow

### Local Development
```bash
cd docs
npm start
```

### Production Build
```bash
cd docs
npm run build
npm run deploy
```

## Troubleshooting

### Build Errors
- Ensure all dependencies are installed: `npm install`
- Check for TypeScript errors: `npm run build`
- Verify all imports are correct

### Deployment Issues
- Check that gh-pages branch was created
- Verify GitHub Pages settings
- Ensure repository is public (required for free GitHub Pages)

### URL Issues
- Update all hardcoded URLs to match your repository
- Check that homepage field in package.json is correct
- Verify GitHub Pages is enabled in repository settings

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to the `docs/public/` directory with your domain
2. Configure DNS settings to point to GitHub Pages
3. Enable custom domain in GitHub Pages settings

## Automated Deployment

For automated deployment on every push to main:

1. Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '18'
        
    - name: Install dependencies
      run: |
        cd docs
        npm install
        
    - name: Build
      run: |
        cd docs
        npm run build
        
    - name: Deploy
      run: |
        cd docs
        npm run deploy
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

2. Push to main branch to trigger deployment

## Features

The deployed React app includes:

- **Home Page**: Project overview and quick start
- **Architecture**: Detailed architectural justification and analysis
- **Results**: Interactive charts and performance metrics
- **Implementation**: Step-by-step implementation guide
- **Interactive Demo**: Live forecasting demo with controls

## Maintenance

- Update content by editing files in `docs/src/pages/`
- Rebuild and redeploy after changes
- Monitor GitHub Pages status in repository settings
- Check for broken links and update URLs as needed
