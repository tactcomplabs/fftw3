#include <stdio.h>
#include <string.h>
#include <stdint.h>

int main() {
    const uint64_t maxP2 = 65536ull; // max VLEN supported by hardware  
    const char* FILE_NAME = "vtw.h";
    FILE* f = fopen(FILE_NAME, "w");
    
    // Base cases
    fputs("/* automatically created by vtw_gen */ \n#ifdef USE_VTW1\n", f);
    fputs("#if VLEN == 1\n#define VTW1(v,x) {TW_CEXP, v, x}\n#endif\n", f);
    fputs("#if VLEN == 2\n#define VTW1(v,x) {TW_CEXP, v, x}, {TW_CEXP, v, x+1}\n#endif\n", f);

    const uint64_t BUFF_SIZE = 100;
    char buffer[BUFF_SIZE];
    char* buff_ptr = &buffer[0]; // silence warning about array decay
    const char* ENDIF = "#endif\n";
    uint64_t currP2 = 4;
    uint64_t i;

    while (currP2 <= maxP2) {
        snprintf(buff_ptr, BUFF_SIZE, "#if VLEN == %llu \n#define VTW1(v, x) \\\n", currP2);
        fputs(buff_ptr, f);
        for (i = 0; i < currP2 - 4; i += 4) {
            snprintf(buff_ptr, BUFF_SIZE, "{TW_CEXP, v, x+%llu}, {TW_CEXP, v, x+%llu}, {TW_CEXP, v, x+%llu}, {TW_CEXP, v, x+%llu} \\\n", i, i+1, i+2, i+3);
            fputs(buff_ptr, f);
        }
        // Do last set separately, as to remove final newline backslash
        snprintf(buff_ptr, BUFF_SIZE, "{TW_CEXP, v, x+%llu}, {TW_CEXP, v, x+%llu}, {TW_CEXP, v, x+%llu}, {TW_CEXP, v, x+%llu} \n", i, i+1, i+2, i+3);
        fputs(buff_ptr, f);        
        fputs(ENDIF, f);
        currP2 <<= 1;
    }

    fputs(ENDIF, f); // close out #ifdef
    fclose(f);
}