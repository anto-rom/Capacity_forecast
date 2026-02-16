from corporate_hybrid_forecast_v17_2 import main

def run():
    out = main()
    run_log = out["run_log"]
    print(run_log.to_string(index=False))

if __name__ == "__main__":
    run()

