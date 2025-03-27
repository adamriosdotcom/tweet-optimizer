# Tweet Optimization System

A comprehensive system for optimizing tweets using reinforcement learning and advanced NLP techniques.

## Features

- **Tweet Optimization**: Automatically optimize tweets for better engagement
- **Engagement Prediction**: Predict engagement scores for tweets
- **Topic Analysis**: Extract and analyze topics from tweets
- **Performance Monitoring**: Track system performance and metrics
- **Historical Analysis**: Analyze historical tweet data for insights

## System Architecture

```
├── tweet_api.py          # FastAPI backend server
├── tweet_rl.py          # Reinforcement learning model
├── database.py          # Database operations
├── process_tweets.py    # Tweet processing pipeline
├── feedback_collector.py # Feedback collection system
├── train_rl_model.py    # Model training script
└── init_db.py          # Database initialization
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

4. Initialize the database:
```bash
python init_db.py
```

## Usage

### Starting the API Server

```bash
python tweet_api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Predict Tweet Engagement
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your tweet text here"}'
```

#### Optimize Tweet
```bash
curl -X POST "http://localhost:8000/optimize" \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your tweet text here", "iterations": 3, "variations": 5}'
```

### Training the Model

```bash
python train_rl_model.py
```

### Collecting Feedback

```bash
python feedback_collector.py
```

## Dashboard (Coming Soon)

A web-based dashboard is under development to provide:
- Tweet optimization interface
- Performance metrics visualization
- Historical data analysis
- System monitoring

## Development

### Running Tests
```bash
pytest
```

### Code Style
```bash
# Install pre-commit hooks
pre-commit install

# Run linting
flake8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

