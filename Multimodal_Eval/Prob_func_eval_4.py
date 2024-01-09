from process import main
from process_base import main_base

if __name__ == "__main__":
    random_seeds = range(40, 50)
    main(random_seeds)
    main_base(random_seeds)