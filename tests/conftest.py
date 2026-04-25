# Load .env before any test module is collected so the SUPABASE_DB_URL
# skipif markers see the variable when pytest imports test files.
import nba_predictor.config  # noqa: F401
