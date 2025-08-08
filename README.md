# Community Vaccination Tracker

A comprehensive system for tracking community vaccination coverage, visualizing data, and managing vaccination records.

![Dashboard Screenshot]-https://ibb.co/60zb1xtT

## Features

- 📊 Real-time vaccination dashboard with summary statistics
- 📍 Geographic visualization of vaccination coverage
- 📈 Detailed reports by age group, gender, and literacy status
- 📝 Record new vaccinations with multi-dose support
- 🗄️ Admin panel for record management
- 📉 Outbreak risk simulation based on coverage data
- 💬 Vaccination hesitancy feedback collection
- 📤 Data export functionality (CSV)

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS (Tailwind), JavaScript (Chart.js, Leaflet)
- **Database**: MongoDB
- **Deployment**: Docker

## Prerequisites

- Docker and Docker Compose
- Python 3.8+

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/community-vaccination-tracker.git
   cd community-vaccination-tracker
   ```

2. Start the services using Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. Access the application at:
   - Frontend: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Configuration

Environment variables can be configured in a `.env` file:

```ini
MONGODB_URI=mongodb://admin:admin@mongo:27017
DB_NAME=vaccine_tracker
COLLECTION_NAME=records
ADMIN_PASSWORD=your_secure_password
```

## API Endpoints

Key endpoints include:

- `POST /submit` - Submit vaccination record
- `GET /records` - Get all records
- `GET /api/dashboard-data` - Dashboard summary data
- `GET /api/report-data` - Report data for visualizations
- `POST /submit-hesitancy` - Submit hesitancy feedback
- `DELETE /delete/{record_id}` - Admin delete record

## Project Structure

```
community-vaccination-tracker/
├── docker-compose.yml        # Docker configuration
├── main.py                   # FastAPI application
├── models.py                 # Pydantic models
├── database.py               # MongoDB connection
├── utils.py                  # Helper functions
├── disease_data.py           # Disease-vaccine mapping
├── requirements.txt          # Python dependencies
└── frontend.html             # Frontend interface
```

## Development

To run without Docker:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start MongoDB (or use a cloud instance)

3. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

**Note**: This is a demo application. For production use, ensure proper security measures are implemented, especially for:
- Database credentials
- Admin operations
- Rate limiting
- Data validation
