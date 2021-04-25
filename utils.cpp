#include "CNN.h"

void load_fm(float* fm, layer l)
{
    char nstr[50];

    sprintf(nstr, "../blobs/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(fm, 1, l.ow*l.oh*l.oc * sizeof(float), fp);
    fclose(fp);
}

void load_weight(float *weight, int length)
{
    char nstr[50];
    sprintf(nstr, "../weights/SkyNet.wt");
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, length*sizeof(float), fp);
    fclose(fp);
}

void load_dwconv_weight(float *weight, layer l)
{
    char nstr[50];
    sprintf(nstr, "../weights/%s.wt", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, l.oc*l.k*l.k*sizeof(float), fp);
    fclose(fp);
}

void load_pwconv_weight(float *weight, layer l)
{
    char nstr[50];
    sprintf(nstr, "../weights/%s.wt", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, l.ic*l.oc*l.k*l.k*sizeof(float), fp);
    fclose(fp);
}

void load_bias(float *bias, layer l)
{
    char nstr[50];
    sprintf(nstr, "../weights/%s.bs", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(bias, 1, l.oc*sizeof(float), fp);
    fclose(fp);
}

void show_fm(float* fm, layer l)
{
    for (int c=0;c<l.oc;c++)
    {
        for (int h=0;h<l.oh;h++)
        {
            for (int w=0;w<l.ow;w++)
            {
                int i = c*l.oh*l.ow + h*l.ow + w;
                std::cout << fm[i]<<", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void check_fm(float* fm, layer l)
{
    int len = l.oc*l.ow*l.oh;
    float *tmp = (float *)malloc(sizeof(float)*len);

    char nstr[50];
    sprintf(nstr, "../blobs/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(tmp, 1, len*sizeof(float), fp);
    fclose(fp);

    int err = 0;

    for (int j = 0; j < len; j++)
    {
        if (((fm[j] - tmp[j]) > check_scale) || ((fm[j] - tmp[j]) < -check_scale))
        {
            err++;
            //printf("[%d] correct=%f,wrong=%f\n", j, tmp[j], fm[j]);
        }
    }

    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);

    free(tmp);
}