#!/bin/bash
# start.sh - Script de démarrage

cd /workspace/InfiniteTalk

# Exécuter le setup si nécessaire
if [ ! -d "/workspace/weights" ]; then
    bash setup_infinitetalk.sh
fi

# Lancer l'API
python api_server.py
