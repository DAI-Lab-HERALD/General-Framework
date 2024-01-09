from process import main
from process_base import main_base

if __name__ == "__main__":
    random_seeds = range(50,60)
    main(random_seeds)
    main_base(random_seeds)