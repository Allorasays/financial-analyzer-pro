# Migration Guide: Secrets Manager → Parameter Store

## Overview

This migration simplifies the infrastructure by replacing AWS Secrets Manager with AWS Systems Manager Parameter Store, saving ~$2/month while maintaining security.

## Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| **Secret Storage** | AWS Secrets Manager | Parameter Store |
| **Cost** | ~$2/month | $0/month |
| **Encryption** | KMS key required | Default AWS key (free) |
| **Rotation** | Lambda automation | Manual (not needed) |
| **Complexity** | High | Low |
| **API** | `get_secret_value()` | `get_parameter()` |

## Migration Steps

### Step 1: Backup Current Secret (If Already Deployed)

```bash
# If you have an existing secret in Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id career_planner/college_scorecard_api_key \
  --query SecretString --output text > backup_secret.json
```

### Step 2: Update Terraform Configuration

**Option A: Use Simplified Terraform Files**

```bash
# Backup current Terraform files
cd terraform
cp main.tf main.tf.backup
cp variables.tf variables.tf.backup

# Use the simplified version
cp main_simplified.tf main.tf
cp outputs_simplified.tf outputs.tf
```

**Option B: Manual Update**

1. Replace `aws_secretsmanager_secret` with `aws_ssm_parameter`
2. Remove Lambda rotation resources
3. Remove KMS key (or keep optional)
4. Update IAM policies to use `ssm:GetParameter`

### Step 3: Deploy Updated Infrastructure

```bash
cd terraform

# Review changes
terraform plan

# Apply changes
terraform apply
```

**Note:** If you have existing Secrets Manager resources, Terraform will:
- Destroy Secrets Manager secret
- Create Parameter Store parameter
- Destroy Lambda function (if not used elsewhere)
- Update IAM policies

### Step 4: Upload Secret to Parameter Store

```bash
# Use the new upload script
python scripts/upload_secret_to_ssm.py

# Or manually
aws ssm put-parameter \
  --name "/career_planner/college_scorecard_api_key" \
  --value "YOUR_API_KEY" \
  --type SecureString \
  --description "College Scorecard API key"
```

### Step 5: Update Application Configuration

The application code has already been updated to use Parameter Store. Just update the parameter name format:

**Old (Secrets Manager):**
```python
college_scorecard_secret_name: str = "career_planner/college_scorecard_api_key"
```

**New (Parameter Store):**
```python
college_scorecard_secret_name: str = "/career_planner/college_scorecard_api_key"
```

The leading `/` is required for Parameter Store paths.

### Step 6: Verify Migration

```bash
# Test parameter retrieval
python -c "from app.integrations.college_scorecard import get_scorecard_api_key; print('API key retrieved:', get_scorecard_api_key()[:10] + '...')"

# Test API endpoint
curl http://localhost:8000/health
```

### Step 7: Clean Up (Optional)

After verifying everything works:

```bash
# Remove old Secrets Manager secret (if not destroyed by Terraform)
aws secretsmanager delete-secret \
  --secret-id career_planner/college_scorecard_api_key \
  --force-delete-without-recovery

# Remove Lambda function (if not needed)
aws lambda delete-function --function-name career-planner-secret-rotation
```

## Code Changes

### Application Code

**Before:**
```python
import boto3
client = boto3.client('secretsmanager')
resp = client.get_secret_value(SecretId=secret_name)
data = json.loads(resp['SecretString'])
api_key = data['COLLEGE_SCORECARD_API_KEY']
```

**After:**
```python
import boto3
ssm = boto3.client('ssm')
resp = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
api_key = resp['Parameter']['Value']
```

### IAM Policy Changes

**Before (Secrets Manager):**
```json
{
  "Effect": "Allow",
  "Action": [
    "secretsmanager:GetSecretValue",
    "secretsmanager:DescribeSecret"
  ],
  "Resource": "arn:aws:secretsmanager:*:*:secret:career_planner/*"
}
```

**After (Parameter Store):**
```json
{
  "Effect": "Allow",
  "Action": [
    "ssm:GetParameter",
    "ssm:GetParameters"
  ],
  "Resource": "arn:aws:ssm:*:*:parameter/career_planner/*"
}
```

## Rollback Plan

If you need to rollback:

1. Restore old Terraform files:
   ```bash
   cd terraform
   mv main.tf.backup main.tf
   mv variables.tf.backup variables.tf
   ```

2. Recreate Secrets Manager secret:
   ```bash
   aws secretsmanager create-secret \
     --name career_planner/college_scorecard_api_key \
     --secret-string file://college_scorecard_api_key.json
   ```

3. Revert application code (check git history)

4. Apply old Terraform:
   ```bash
   terraform apply
   ```

## Benefits of Migration

✅ **Cost Savings**: $0/month vs ~$2/month (100% reduction)
✅ **Simplicity**: Fewer resources to manage
✅ **Same Security**: Encryption at rest and in transit
✅ **Same Functionality**: Retrieve secrets in code
✅ **Faster Setup**: No Lambda rotation complexity
✅ **Easier Debugging**: Simpler parameter management

## What's Lost

❌ **Automatic Rotation**: No built-in rotation (but not needed for API keys)
❌ **Version History**: Limited versioning (but adequate for most use cases)
❌ **Advanced Features**: No automatic expiry, replication (not needed here)

## Verification Checklist

- [ ] Parameter created in Parameter Store
- [ ] Application can retrieve API key
- [ ] API endpoints work correctly
- [ ] Health check passes
- [ ] IAM permissions updated
- [ ] Old Secrets Manager resources removed
- [ ] Lambda rotation removed (if not needed)
- [ ] Costs reduced (check AWS billing)

## Support

If you encounter issues during migration:

1. Check CloudWatch Logs for errors
2. Verify IAM permissions
3. Test parameter retrieval manually:
   ```bash
   aws ssm get-parameter --name "/career_planner/college_scorecard_api_key" --with-decryption
   ```
4. Check application logs

## Cost Comparison

| Service | Before | After | Savings |
|---------|--------|-------|---------|
| Secrets Manager | $0.40/month | $0 | $0.40 |
| KMS Key | $1.00/month | $0* | $1.00 |
| Lambda (Rotation) | $0.30/month | $0 | $0.30 |
| CloudWatch Logs | $0.05/month | $0.05 | $0 |
| **Total** | **~$1.75/month** | **~$0.05/month** | **~$1.70/month** |

*Using default AWS managed key (free). Custom KMS adds $1/month if needed.

## Next Steps After Migration

1. ✅ Update documentation
2. ✅ Update CI/CD pipelines (if using Parameter Store in deployments)
3. ✅ Train team on new parameter management
4. ✅ Monitor costs to confirm savings
5. ✅ Remove old backup files

---

**Migration Status:** Ready to migrate
**Estimated Time:** 1-2 hours
**Risk Level:** Low (easy rollback)
**Recommended:** Yes

