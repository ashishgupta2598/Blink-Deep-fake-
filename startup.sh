pm2 start redis_smileBlink_worker.py --interpreter python3 --name "ai-Blink-ManagedBlinkModels"

gunicorn -c redis_streamer_gunicorn.py app:api

