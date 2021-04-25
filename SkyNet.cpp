#include "CNN.h"

layer config[layer_count] = {
{ "data",    320,160,3,  320,160,32, 0,0,0},    //data
{ "dwconv1", 320,160,32, 320,160,32, 3,1,1},    //dwconv1
{ "pwconv1", 320,160,32, 320,160,64, 1,1,0},    //pwconv1
{ "pool1",   320,160,64, 160,80,64,  2,2,0},    //pool1
{ "dwconv2", 160,80,64,  160,80,64,  3,1,1},    //dwconv2
{ "pwconv2", 160,80,64,  160,80,96,  1,1,0},   //pwconv2
{ "pool2",   160,80,96,  80,40,96,   2,2,0},    //pool2
{ "dwconv3", 80,40,96,   80,40,96,   3,1,1},    //dwconv3
{ "pwconv3", 80,40,96,   80,40,192,  1,1,0},    //pwconv3
{ "reorg",   80,40,192,  40,20,768,  2,2,0},    //reorg
{ "pool3",   80,40,192,  40,20,192,  2,2,0},    //pool3
{ "dwconv4", 40,20,192,  40,20,192,  3,1,1},    //dwconv4
{ "pwconv4", 40,20,192,  40,20,384,  1,1,0},    //pwconv4
{ "dwconv5", 40,20,384,  40,20,384,  3,1,1},    //dwconv5
{ "pwconv5", 40,20,384,  40,20,512,  1,1,0},    //pwconv5
{ "cat",     40,20,192,  40,20,1280, 0,0,0},    //concat
{ "dwconv6", 40,20,1280, 40,20,1280, 3,1,1},    //dwconv6
{ "pwconv6", 40,20,1280, 40,20,96,   1,1,0},    //pwconv6
{ "conv7",   40,20,96,   40,20,10,   1,1,0},    //conv7
};

DT* data_blob;
DT* parameter;
DT* blob[2];
DT* reorg_blob;
DT* weight;
DT* bias;
void SkyNet_init()
{
    parameter = (DT*)sds_alloc(442634*sizeof(DT));
    blob[0] = (DT*)sds_alloc(3276800*sizeof(DT));
    blob[1] = (DT*)sds_alloc(3276800*sizeof(DT));
    reorg_blob = (DT*)sds_alloc(config[9].oc*config[9].oh*config[9].ow*sizeof(DT));
    load_weight(parameter, 442634);
    weight = (DT*)sds_alloc(64*96*sizeof(DT));
    bias = (DT*)sds_alloc(96*sizeof(DT));

}

void Load_FM(DT* ifm, DT IFM[32][43][83], int Hx, int Wx, int Cx, int ow, int oh)
{
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
#pragma HLS PIPELINE II=1
            int ifm_index = ;
            for (int c=0; c<32; c++)
            {
                IFM[c][h][w] = ifm[ifm_index];
            }
        }
    }
}

void Load_WBUF3x3(DT* weight, DT WBUF3x3[32][3][3], int Mx, layer l)
{
    for(int m=0; m<3; m++)
    {
        for(int n=0; n<3; n++)
        {
            for(int c=0; c<32; c++)
            {
                WBUF3x3[c][m][n] = weight[ ];
            }
        }
    }
}

void Add_Bias(DT FM[32][43][83], DT BBUF[32], int relu)
{
    for(int h = ; h < ; h++){
        for(int w = ; w< ; w++){
            for(int c=0; c<32; c++){
                DT odata = FM[c][h][w];
                odata += BBUF[c];
                if(odata<0)
                    FM[c][h][w] = 0;
                else
                    FM[c][h][w] = odata;
            }
        }
    }
}

void Load_BBUF(DT* bias, DT BBUF[32], int Mx)
{
    for(int c=0; c<32; c++)
    {
        BBUF[c] = bias[ ];
    }
}

void Export_FM(DT* fm, DT OFM[32][43][83], int Hx, int Wx, int Cx, int ow, int oh)
{
    for (int h= ; h< ; h++)
    {
        for (int w= ; w< ; w++)
        {
#pragma HLS PIPELINE II=1
            int fm_index = ;
            fm[fm_index] = OFM[c][h][w];
        }
    }
}

void DWCONV3X3(DT IFM[32][43][83], DT OFM[32][43][83], DT WBUF3x3[32][3][3])
{
    for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			for(int h= ; h< ; h++){
				for(int w= ; w< ; w++){
					for(int c=0; c<32; c++){
						OFM[c][h][w] += WBUF3x3[c][i][j] * IFM[c][h+i-1][w+j-1];
					}
				}
			}
		}
	}
}

void PWCONV1X1(DT IFM[32][43][83], DT OFM[32][43][83], DT WBUF1x1[32][32])
{
	for(int h= ; h< ; h++){
		for(int w= ; w<= ; w++){
			for (int tm=0; tm<32; tm++){
				DT odatatmp = OFM[tm][h][w];
				DT odata = 0;
				for (int tn=0; tn<32; tn++){
					odata += WBUF1x1[tm][tn]*IFM[tn][h][w];
				}
                OFM[tm][h][w] = odata + odatatmp;
			}
		}
	}
}

void SkyNet()
{
    int input = 0;
    load_fm(blob[input], config[0]);
  
    int weight_offset = 0;
    int bias_offset = weight_offset + config[1].oc*config[1].k*config[1].k;
    /*******dwconv1*******/
    dwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[1]);
    
    for(int Hx=0; Hx<2; Hx++)
    {
        for(int Wx=0; Wx<2; Wx++)
        {
            Load_FM();
            Load_WBUF3x3();
            Load_BBUF(&parameter[bias_offset], );
            DWCONV3X3(IFM, OFM, WBUF3x3);
            Add_Bias();
            Export_FM();
        }
    }
    check_fm(blob[1-input], config[1]);

    input = 1 - input;
    /*******pwconv1*******/
    load_fm(blob[input], config[1]);
    weight_offset = bias_offset + config[1].oc;
    bias_offset = weight_offset + config[2].oc*config[2].ic*config[2].k*config[2].k;
    pwconv(blob[input], blob[1-input], &parameter[weight_offset], &parameter[bias_offset], 1, config[2]);
    check_fm(blob[1-input], config[2]);
    input = 1 - input;
    /*******pool1*******/
    maxpool(blob[input], blob[1-input], config[3]);
    check_fm(blob[1-input], config[3]);
    input = 1 - input;
}