#!/usr/bin/env python3

import datetime
import logging
import os
import subprocess
import sys

import boto3
import psycopg2
import redis
from botocore.exceptions import ClientError
from elasticsearch import Elasticsearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/var/log/backups/backup.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("backup-manager")


class BackupManager:
    def __init__(self):
        """Initialize backup manager with environment variables."""
        self.aws_bucket = os.environ["BACKUP_BUCKET"]
        self.s3_client = boto3.client("s3")

        # Database connection info
        self.db_host = os.environ["POSTGRES_HOST"]
        self.db_name = os.environ["POSTGRES_DB"]
        self.db_user = os.environ["POSTGRES_USER"]
        self.db_password = os.environ["POSTGRES_PASSWORD"]

        # Elasticsearch connection info
        self.es_host = os.environ["ELASTICSEARCH_HOST"]
        self.es_port = os.environ["ELASTICSEARCH_PORT"]

        # Redis connection info
        self.redis_host = os.environ["REDIS_HOST"]
        self.redis_port = os.environ["REDIS_PORT"]

        # Timestamp for backup files
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def backup_database(self):
        """Backup PostgreSQL database."""
        try:
            backup_file = f"/tmp/db_backup_{self.timestamp}.sql"

            # Create database dump
            cmd = [
                "pg_dump",
                "-h",
                self.db_host,
                "-U",
                self.db_user,
                "-d",
                self.db_name,
                "-F",
                "c",  # Custom format
                "-f",
                backup_file,
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = self.db_password

            subprocess.run(cmd, env=env, check=True)

            # Upload to S3
            s3_key = f"database/db_backup_{self.timestamp}.sql"
            self.upload_to_s3(backup_file, s3_key)

            # Cleanup
            os.remove(backup_file)
            logger.info(f"Database backup completed: {s3_key}")

        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            raise

    def backup_elasticsearch(self):
        """Create Elasticsearch snapshot."""
        try:
            es = Elasticsearch([f"http://{self.es_host}:{self.es_port}"])

            # Register repository if not exists
            repo_name = "legal_search_backup"
            if not es.snapshot.verify_repository(repository=repo_name):
                es.snapshot.create_repository(
                    repository=repo_name,
                    body={
                        "type": "s3",
                        "settings": {
                            "bucket": self.aws_bucket,
                            "base_path": "elasticsearch",
                        },
                    },
                )

            # Create snapshot
            snapshot_name = f"snapshot_{self.timestamp}"
            es.snapshot.create(
                repository=repo_name, snapshot=snapshot_name, wait_for_completion=True
            )

            logger.info(f"Elasticsearch snapshot completed: {snapshot_name}")

        except Exception as e:
            logger.error(f"Elasticsearch backup failed: {str(e)}")
            raise

    def backup_redis(self):
        """Backup Redis data."""
        try:
            r = redis.Redis(host=self.redis_host, port=int(self.redis_port))

            # Save current state
            r.save()

            # Get dump file
            dump_file = "/var/lib/redis/dump.rdb"
            backup_file = f"/tmp/redis_backup_{self.timestamp}.rdb"

            # Copy dump file
            subprocess.run(["cp", dump_file, backup_file], check=True)

            # Upload to S3
            s3_key = f"redis/redis_backup_{self.timestamp}.rdb"
            self.upload_to_s3(backup_file, s3_key)

            # Cleanup
            os.remove(backup_file)
            logger.info(f"Redis backup completed: {s3_key}")

        except Exception as e:
            logger.error(f"Redis backup failed: {str(e)}")
            raise

    def upload_to_s3(self, file_path, s3_key):
        """Upload file to S3 bucket."""
        try:
            self.s3_client.upload_file(file_path, self.aws_bucket, s3_key)
        except ClientError as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise

    def cleanup_old_backups(self):
        """Remove old backups based on retention policy."""
        try:
            # List objects in bucket
            for prefix in ["database/", "redis/"]:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.aws_bucket, Prefix=prefix
                )

                if "Contents" in response:
                    # Sort by date
                    objects = sorted(
                        response["Contents"], key=lambda x: x["LastModified"]
                    )

                    # Keep last 7 days of backups
                    retention_days = 7
                    cutoff_date = datetime.datetime.now(
                        datetime.timezone.utc
                    ) - datetime.timedelta(days=retention_days)

                    # Delete old backups
                    for obj in objects:
                        if obj["LastModified"] < cutoff_date:
                            self.s3_client.delete_object(
                                Bucket=self.aws_bucket, Key=obj["Key"]
                            )
                            logger.info(f'Deleted old backup: {obj["Key"]}')

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise

    def run_backup(self):
        """Run all backup operations."""
        try:
            logger.info("Starting backup process...")

            # Run backups
            self.backup_database()
            self.backup_elasticsearch()
            self.backup_redis()

            # Cleanup old backups
            self.cleanup_old_backups()

            logger.info("Backup process completed successfully")

        except Exception as e:
            logger.error(f"Backup process failed: {str(e)}")
            sys.exit(1)


def main():
    """Main entry point for backup script."""
    try:
        backup_manager = BackupManager()
        backup_manager.run_backup()
    except Exception as e:
        logger.error(f"Backup script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
