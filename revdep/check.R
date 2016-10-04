
library("devtools")

revdep_check(threads = 3)
revdep_check_save_summary()
revdep_check_print_problems()