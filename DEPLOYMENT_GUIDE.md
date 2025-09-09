# 🚀 Production Deployment Guide

## Render.com Deployment Steps

### 1. Prepare Your Repository
```bash
# Make sure all files are committed
git add .
git commit -m "Ready for production deployment"
git push origin main
```

### 2. Deploy on Render.com

1. **Go to [render.com](https://render.com)**
2. **Sign up/Login with GitHub**
3. **Click "New +" → "Blueprint"**
4. **Connect your GitHub repository**
5. **Render will automatically detect your `render.yaml`**

### 3. Environment Variables

Set these in Render dashboard:

#### For API Service:
- `SECRET_KEY`: `your-super-secret-key-change-this`
- `API_BASE_URL`: `https://financial-analyzer-api.onrender.com` (auto-generated)

#### For Streamlit Service:
- `API_BASE_URL`: `https://financial-analyzer-api.onrender.com` (same as API)

### 4. Custom Domain (Optional)
- Go to your service settings
- Add custom domain in "Custom Domains" section
- Update DNS records as instructed

### 5. Monitoring
- Check logs in Render dashboard
- Monitor performance metrics
- Set up alerts for downtime

## 🔧 Post-Deployment Checklist

- [ ] API service is running (check health endpoint)
- [ ] Streamlit app loads correctly
- [ ] API calls work from frontend
- [ ] All features function properly
- [ ] Performance is acceptable
- [ ] SSL certificate is active

## 🆘 Troubleshooting

### Common Issues:
1. **Build fails**: Check `requirements.txt` and Python version
2. **API not accessible**: Verify CORS settings and URLs
3. **Streamlit not loading**: Check port configuration
4. **Slow performance**: Consider upgrading to paid plan

### Debug Commands:
```bash
# Check API health
curl https://your-api-url.onrender.com/health

# Check API endpoints
curl https://your-api-url.onrender.com/
```

## 📊 Performance Optimization

### Free Tier Limitations:
- Services sleep after 15 minutes of inactivity
- Cold start can take 30-60 seconds
- Limited CPU and memory

### Upgrade Recommendations:
- **Starter Plan ($7/month)**: Always-on services
- **Standard Plan ($25/month)**: Better performance
- **Pro Plan ($85/month)**: Production-ready

## 🔒 Security Considerations

1. **Change default SECRET_KEY**
2. **Enable HTTPS only**
3. **Set up proper CORS policies**
4. **Add rate limiting**
5. **Monitor for suspicious activity**

## 📈 Scaling Strategy

1. **Start with free tier** for testing
2. **Upgrade to paid** for production use
3. **Add database** for user data
4. **Implement caching** for better performance
5. **Add CDN** for static assets

---

**Your app will be live at:**
- Frontend: `https://financial-analyzer-streamlit.onrender.com`
- API: `https://financial-analyzer-api.onrender.com`
