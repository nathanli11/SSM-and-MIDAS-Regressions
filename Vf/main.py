import data_management
import MC_SIMUL
import SSM_MIDAS_OOS
import SSM_MIDAS_FORECAST

RUN_DATA = True
RUN_KALMAN = True
RUN_MC = False
RUN_TABLES = True

def main():
    if RUN_DATA:
        print("=== DATA MANAGEMENT ===")
        data = data_management.main()

    if RUN_KALMAN:
        print("=== SSM MIDAS FORECAST ===")
        SSM_MIDAS_FORECAST.main()

    if RUN_MC:
        print("=== MONTE CARLO SIMULATIONS ===")
        MC_SIMUL.main()

    if RUN_TABLES:
        print("=== SSM MIDAS OOS ===")
        SSM_MIDAS_OOS.main()

if __name__ == "__main__":
    main()
