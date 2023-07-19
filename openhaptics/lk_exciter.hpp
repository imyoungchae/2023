#include <signal.h>
static volatile sig_atomic_t flag = 0;

void callback(int sig) {

    flag = 1;
}

class lk_exciter {

    public:
        bool ok();
        lk_exciter() { signal(SIGINT, callback); }
};

bool lk_exciter::ok() {
    
    return flag==0;
}