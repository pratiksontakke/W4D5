# Deployment Guide

## Overview

This guide covers the deployment process for the Indian Legal Document Search System. The system is containerized using Docker and deployed on AWS ECS for scalability and reliability.

## Prerequisites

1. AWS Account with appropriate permissions
2. Docker installed locally
3. AWS CLI configured
4. Access to container registry (DockerHub)
5. Domain name and SSL certificates

## Infrastructure Setup

### 1. AWS Resources

```bash
# Create VPC and networking
aws cloudformation create-stack \
    --stack-name legal-search-network \
    --template-body file://infrastructure/network.yaml

# Create ECS cluster
aws ecs create-cluster --cluster-name legal-search

# Create ECR repository
aws ecr create-repository --repository-name legal-search
```

### 2. Database Setup

```bash
# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier legal-search \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --allocated-storage 100
```

### 3. Elasticsearch Setup

```bash
# Create Elasticsearch domain
aws elasticsearch create-elasticsearch-domain \
    --domain-name legal-search \
    --elasticsearch-version 7.10
```

## Application Deployment

### 1. Build and Push Docker Image

```bash
# Build image
docker build -t legal-search .

# Tag image
docker tag legal-search:latest your-registry/legal-search:latest

# Push to registry
docker push your-registry/legal-search:latest
```

### 2. Deploy to ECS

```bash
# Create task definition
aws ecs register-task-definition \
    --cli-input-json file://deployment/task-definition.json

# Create service
aws ecs create-service \
    --cluster legal-search \
    --service-name api \
    --task-definition legal-search \
    --desired-count 2
```

### 3. Configure Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
    --name legal-search-alb \
    --subnets subnet-1 subnet-2 \
    --security-groups sg-123

# Create target group
aws elbv2 create-target-group \
    --name legal-search-tg \
    --protocol HTTP \
    --port 80 \
    --vpc-id vpc-123
```

## Environment Variables

Set these environment variables in AWS Systems Manager Parameter Store:

```bash
aws ssm put-parameter \
    --name "/legal-search/prod/DATABASE_URL" \
    --value "postgresql://..." \
    --type SecureString

aws ssm put-parameter \
    --name "/legal-search/prod/ELASTICSEARCH_URL" \
    --value "https://..." \
    --type SecureString
```

## Monitoring Setup

### 1. CloudWatch

```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/legal-search

# Create metric filters
aws logs put-metric-filter \
    --log-group-name /ecs/legal-search \
    --filter-name errors \
    --filter-pattern "ERROR" \
    --metric-transformations \
        metricName=ErrorCount,metricNamespace=LegalSearch,metricValue=1
```

### 2. Alerts

```bash
# Create CloudWatch alarm
aws cloudwatch put-metric-alarm \
    --alarm-name legal-search-error-rate \
    --metric-name ErrorCount \
    --namespace LegalSearch \
    --threshold 10 \
    --period 300 \
    --evaluation-periods 2 \
    --comparison-operator GreaterThanThreshold
```

## Backup Configuration

### 1. Database Backups

```bash
# Configure automated backups
aws rds modify-db-instance \
    --db-instance-identifier legal-search \
    --backup-retention-period 7 \
    --preferred-backup-window "03:00-04:00"
```

### 2. Document Backups

```bash
# Create S3 bucket with versioning
aws s3api create-bucket --bucket legal-search-documents
aws s3api put-bucket-versioning \
    --bucket legal-search-documents \
    --versioning-configuration Status=Enabled
```

## Security Measures

### 1. SSL/TLS Configuration

```bash
# Request certificate
aws acm request-certificate \
    --domain-name api.legal-search.com \
    --validation-method DNS

# Configure ALB listener
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=$CERT_ARN
```

### 2. Security Groups

```bash
# Create security group
aws ec2 create-security-group \
    --group-name legal-search-sg \
    --description "Legal Search security group"

# Configure rules
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0
```

## Scaling Configuration

### 1. Auto Scaling

```bash
# Create Auto Scaling group
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/legal-search/api \
    --min-capacity 2 \
    --max-capacity 10

# Configure scaling policy
aws application-autoscaling put-scaling-policy \
    --policy-name cpu-tracking \
    --service-namespace ecs \
    --resource-id service/legal-search/api \
    --scalable-dimension ecs:service:DesiredCount \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

### 2. Database Scaling

```bash
# Configure read replicas
aws rds create-db-instance-read-replica \
    --db-instance-identifier legal-search-replica \
    --source-db-instance-identifier legal-search
```

## Rollback Procedure

In case of deployment issues:

```bash
# Rollback to previous task definition
aws ecs update-service \
    --cluster legal-search \
    --service api \
    --task-definition legal-search:$PREVIOUS_VERSION

# Wait for deployment
aws ecs wait services-stable \
    --cluster legal-search \
    --services api
```

## Maintenance Windows

Configure maintenance windows in AWS Systems Manager:

```bash
# Create maintenance window
aws ssm create-maintenance-window \
    --name "legal-search-maintenance" \
    --schedule "cron(0 2 ? * SUN *)" \
    --duration 2 \
    --cutoff 1

# Register targets
aws ssm register-target-with-maintenance-window \
    --window-id $WINDOW_ID \
    --targets "Key=tag:Environment,Values=production" \
    --owner-information "Legal Search System"
```

## Troubleshooting

1. Check application logs:
```bash
aws logs get-log-events \
    --log-group-name /ecs/legal-search \
    --log-stream-name $LOG_STREAM
```

2. Check service health:
```bash
aws ecs describe-services \
    --cluster legal-search \
    --services api
```

3. Monitor metrics:
```bash
aws cloudwatch get-metric-statistics \
    --namespace AWS/ECS \
    --metric-name CPUUtilization \
    --dimensions Name=ClusterName,Value=legal-search \
    --start-time $START \
    --end-time $END \
    --period 300 \
    --statistics Average
```

## Support Contacts

- DevOps Team: devops@legal-search.com
- Infrastructure Team: infra@legal-search.com
- Security Team: security@legal-search.com

## Compliance Requirements

1. Data Retention: 7 years
2. Backup Frequency: Daily
3. Security Audits: Quarterly
4. Penetration Testing: Bi-annual
5. Compliance Reports: Monthly
