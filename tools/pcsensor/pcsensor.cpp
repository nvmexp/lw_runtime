/*
 * pcsensor.c by Michitaka Ohno (c) 2011 (elpeo@mars.dti.ne.jp)
 * based oc pcsensor.c by Juan Carlos Perez (c) 2011 (cray@isp-sl.com)
 * based on Temper.c by Robert Kavaler (c) 2009 (relavak.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 * THIS SOFTWARE IS PROVIDED BY Juan Carlos Perez ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Robert kavaler BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */

#include "pcsensor.h"
#include "core/include/tee.h"
#include "core/include/xp.h"

#define INTERFACE1 (0x00)
#define INTERFACE2 (0x01)
#define SUPPORTED_DEVICES (2)

bool libUsbInitialized = false;

typedef int (*fn_detach_kernel_driver_np)(usb_dev_handle *, int);
typedef usb_dev_handle * (*fn_open)(struct usb_device *);
typedef void (*fn_set_debug)(int);
typedef void (*fn_init)(void);
typedef int (*fn_find_busses)(void);
typedef int (*fn_find_devices)(void);
typedef usb_device * (*fn_device)(usb_dev_handle *);
typedef int (*fn_set_configuration)(usb_dev_handle *, int);
typedef int (*fn_claim_interface)(usb_dev_handle *, int);
typedef int (*fn_control_msg)(usb_dev_handle *, int, int, int, int, 
                             char *, int, int);
typedef int (*fn_interrupt_read)(usb_dev_handle *, int, char *, int, int);
typedef int (*fn_release_interface)(usb_dev_handle *, int);
typedef int (*fn_close)(usb_dev_handle *);
typedef usb_bus * (*fn_get_busses)(void);

fn_detach_kernel_driver_np  usb_detach_kernel_driver_np;
fn_open                     usb_open;
fn_set_debug                usb_set_debug;
fn_init                     usb_init;
fn_find_busses              usb_find_busses;
fn_find_devices             usb_find_devices;
fn_device                   usb_device;
fn_set_configuration        usb_set_configuration;
fn_claim_interface          usb_claim_interface;
fn_control_msg              usb_control_msg;
fn_interrupt_read           usb_interrupt_read;
fn_release_interface        usb_release_interface;
fn_close                    usb_close;
fn_get_busses               usb_get_busses;

#define GET_LIBUSB_FUNCTION(function_name)                                    \
do {                                                                        \
    usb_##function_name =                                                    \
        (fn_##function_name) Xp::GetDLLProc(libUsbHandle, "usb_"#function_name); \
    if (!usb_##function_name)                                                \
    {                                                                         \
        Printf(Tee::PriLow, "Cannot load libusb function (%s)\n",              \
               #function_name);                                               \
        return RC::DLL_LOAD_FAILED;                                           \
    }                                                                         \
} while (0)

RC PcSensor::InitializeLibUsb()
{
    RC rc;
    
    if (libUsbInitialized)
    {
        return OK;
    }
    
    void *libUsbHandle = nullptr;
    string libUsbName = "libusb" + Xp::GetDLLSuffix();
    rc = Xp::LoadDLL(libUsbName, &libUsbHandle, false);
    
    if (OK != rc)
    {
        Printf(Tee::PriLow, "pcsensor: "
                            "Cannot load %s library required for reading "
                            "from USB temperature sensor. rc = %d\n", 
                            libUsbName.c_str(), (UINT32)rc);
        Printf(Tee::PriLow, "pcsensor: In order to support it, please install "
                            "libusb-0.1\n");
        return rc;
    }
    
    GET_LIBUSB_FUNCTION(detach_kernel_driver_np);
    GET_LIBUSB_FUNCTION(open);
    GET_LIBUSB_FUNCTION(set_debug);
    GET_LIBUSB_FUNCTION(init);
    GET_LIBUSB_FUNCTION(find_busses);
    GET_LIBUSB_FUNCTION(find_devices);
    GET_LIBUSB_FUNCTION(device);
    GET_LIBUSB_FUNCTION(set_configuration);
    GET_LIBUSB_FUNCTION(claim_interface);
    GET_LIBUSB_FUNCTION(control_msg);
    GET_LIBUSB_FUNCTION(interrupt_read);
    GET_LIBUSB_FUNCTION(release_interface);
    GET_LIBUSB_FUNCTION(close);
    GET_LIBUSB_FUNCTION(get_busses);

    libUsbInitialized = true;
    
    return rc;
}


const static unsigned short vendor_id[] = { 
    0x1130,
    0x0c45
};
const static unsigned short product_id[] = { 
    0x660c,
    0x7401
};

const static unsigned char uTemperatura[] = { 0x01, 0x80, 0x33, 0x01, 0x00, 0x00, 0x00, 0x00 };
const static unsigned char uIni1[] = { 0x01, 0x82, 0x77, 0x01, 0x00, 0x00, 0x00, 0x00 };
const static unsigned char uIni2[] = { 0x01, 0x86, 0xff, 0x01, 0x00, 0x00, 0x00, 0x00 };
const static unsigned char uCmd0[] = {    0,    0,    0,    0,    0,    0,    0,    0 };
const static unsigned char uCmd1[] = {   10,   11,   12,   13,    0,    0,    2,    0 };
const static unsigned char uCmd2[] = {   10,   11,   12,   13,    0,    0,    1,    0 };
const static unsigned char uCmd3[] = { 0x52,    0,    0,    0,    0,    0,    0,    0 };
const static unsigned char uCmd4[] = { 0x54,    0,    0,    0,    0,    0,    0,    0 };

const static int reqIntLen=8;
const static int timeout=5000; /* timeout in ms */

static int device_type(usb_dev_handle *lvr_winusb){
    struct usb_device *dev;
    int i;
    if (lvr_winusb == nullptr)
    {
        return -1;
    }
    dev = usb_device(lvr_winusb);
    for(i =0;i < SUPPORTED_DEVICES;i++){
        if (dev->descriptor.idVendor == vendor_id[i] && 
            dev->descriptor.idProduct == product_id[i] ) {
            return i;
        }
    }
    return -1;
}

static int usb_detach(usb_dev_handle *lvr_winusb, int iInterface) {
    int ret;

    ret = usb_detach_kernel_driver_np(lvr_winusb, iInterface);
    if(ret) {
        if(errno == ENODATA) {
            Printf(Tee::PriDebug, "pcsensor: Device already detached\n");
        } else {
            Printf(Tee::PriLow, "pcsensor: Detach failed: %s[%d]\n",
                   strerror(errno), errno);
            Printf(Tee::PriLow, "pcsensor: Continuing anyway\n");
        }
    } else {
        Printf(Tee::PriDebug, "pcsensor: detach successful\n");
    }
    return ret;
} 

static usb_dev_handle *find_lvr_winusb() {
    struct usb_bus *bus;
    struct usb_device *dev;
    int i;

    struct usb_bus *usb_busses = usb_get_busses();

    for (bus = usb_busses; bus; bus = bus->next) {
        for (dev = bus->devices; dev; dev = dev->next) {
            for(i =0;i < SUPPORTED_DEVICES;i++){
                if (dev->descriptor.idVendor == vendor_id[i] && 
                    dev->descriptor.idProduct == product_id[i] ) {
                    usb_dev_handle *handle;

                    Printf(Tee::PriDebug, "pcsensor: "
                                          "lvr_winusb with Vendor Id: %x "
                                          "and Product Id: %x found.\n",
                                          vendor_id[i], product_id[i]);
                    

                    if (!(handle = usb_open(dev))) {
                        Printf(Tee::PriLow, "pcsensor: "
                                             "Could not open USB device\n");
                        return nullptr;
                    }
                    return handle;
                }
            }
        }
    }
    return nullptr;
}

static usb_dev_handle* setup_libusb_access() {
    usb_dev_handle *lvr_winusb;

    // Change to usb_set_debug(255) to generate low level USB debug info
    usb_set_debug(0);

    usb_init();
    usb_find_busses();
    usb_find_devices();

    if(!(lvr_winusb = find_lvr_winusb())) {
        Printf(Tee::PriLow, "pcsensor: Could not find the USB device. "
                             "Aborting USB setup.\n");
        return nullptr;
    }
    
    usb_detach(lvr_winusb, INTERFACE1);
    usb_detach(lvr_winusb, INTERFACE2);
    
    if (usb_set_configuration(lvr_winusb, 0x01) < 0) {
        Printf(Tee::PriLow, "pcsensor: Could not set configuration 1\n");
        return nullptr;
    }

    // Microdia tiene 2 interfaces
    if (usb_claim_interface(lvr_winusb, INTERFACE1) < 0) {
        Printf(Tee::PriLow, "pcsensor: Could not claim interface\n");
        return nullptr;
    }

    if (usb_claim_interface(lvr_winusb, INTERFACE2) < 0) {
        Printf(Tee::PriLow, "pcsensor: Could not claim interface\n");
        return nullptr;
    }

    return lvr_winusb;
}
 
static int ini_control_transfer(usb_dev_handle *dev) {
    int r;

    char question[] = { 0x01,0x01 };
    int questionLength = (sizeof(question)/sizeof(question[0]));

    r = usb_control_msg(dev, 0x21, 0x09, 0x0201,
                        0x00, (char *) question, questionLength, timeout);
    if( r < 0 )
    {
        Printf(Tee::PriLow, "pcsensor: error on USB control write"); 
        return -1;
    }

    Printf(Tee::PriDebug, "pcsensor: ");
    for (int i = 0; i < questionLength; i++)
    {
        Printf(Tee::PriDebug, "%02x ", question[i] & 0xFF);
    }
    Printf(Tee::PriDebug, "\n");
    
    return 0;
}
 
static int control_transfer(usb_dev_handle *dev, const unsigned char *pquestion) {
    int r;

    char question[reqIntLen];
    
    memcpy(question, pquestion, sizeof question);

    r = usb_control_msg(dev, 0x21, 0x09, 0x0200, 0x01,
                        (char *) question, reqIntLen, timeout);
    if( r < 0 )
    {
        Printf(Tee::PriLow, "pcsensor: error on USB control write");
        return -1;
    }

    Printf(Tee::PriDebug, "pcsensor: ");
    for (int i = 0; i < reqIntLen; i++)
    {
        Printf(Tee::PriDebug, "%02x ",question[i]  & 0xFF);
    }
    Printf(Tee::PriDebug, "\n");

    return 0;
}

static int interrupt_read(usb_dev_handle *dev) {
    int r;
    char answer[reqIntLen];
    bzero(answer, reqIntLen);
    
    r = usb_interrupt_read(dev, 0x82, answer, reqIntLen, timeout);
    if( r != reqIntLen )
    {
        Printf(Tee::PriLow, "pcsensor: error on USB interrupt read");
        return -1;
    }

    Printf(Tee::PriDebug, "pcsensor: ");
    for (int i = 0; i < reqIntLen; i++)
    {
        Printf(Tee::PriDebug, "%02x ",answer[i]  & 0xFF);
    }
    Printf(Tee::PriDebug, "\n");

    return 0;
}

static int interrupt_read_temperatura(usb_dev_handle *dev, float *tempC) {
 
    int r, temperature;
    char answer[reqIntLen];
    bzero(answer, reqIntLen);
    
    r = usb_interrupt_read(dev, 0x82, answer, reqIntLen, timeout);
    if( r != reqIntLen )
    {
        Printf(Tee::PriLow, "pcsensor: error on USB interrupt read");
        return -1;
    }

    Printf(Tee::PriDebug, "pcsensor: ");
    for (int i = 0; i < reqIntLen; i++)
    {
        Printf(Tee::PriDebug, "%02x ",answer[i]  & 0xFF);
    }
    Printf(Tee::PriDebug, "\n");
    
    temperature = (answer[3] & 0xFF) + (answer[2] << 8);
    *tempC = temperature * (125.0 / 32000.0);
    return 0;
}

static int get_data(usb_dev_handle *dev, char *buf, int len){
    return usb_control_msg(dev, 0xa1, 1, 0x300, 0x01, (char *)buf, len, timeout);
}

static int get_temperature(usb_dev_handle *dev, float *tempC){
    char buf[256];
    int ret, temperature, i;

    control_transfer(dev, uCmd1 );
    control_transfer(dev, uCmd4 );
    for(i = 0; i < 7; i++) {
        control_transfer(dev, uCmd0 );
    }
    control_transfer(dev, uCmd2 );
    ret = get_data(dev, buf, 256);
    if(ret < 2) {
        return -1;
    }

    temperature = (buf[1] & 0xFF) + (buf[0] << 8);    
    *tempC = temperature * (125.0 / 32000.0);
    return 0;
}

usb_dev_handle* PcSensor::pcsensor_open(){
    usb_dev_handle *lvr_winusb = nullptr;
    char buf[256];
    int ret;

    if (!(lvr_winusb = setup_libusb_access())) {
        return nullptr;
    } 

    switch(device_type(lvr_winusb)){
    case 0:
        control_transfer(lvr_winusb, uCmd1 );
        control_transfer(lvr_winusb, uCmd3 );
        control_transfer(lvr_winusb, uCmd2 );
        ret = get_data(lvr_winusb, buf, 256);
        
        Printf(Tee::PriDebug, "pcsensor: Other Stuff (%d bytes):\n", ret);
        for(int i = 0; i < ret; i++) {
            Printf(Tee::PriDebug, " %02x", buf[i] & 0xFF);
            if(i % 16 == 15) {
                Printf(Tee::PriDebug, "\n");
            }
        }
        Printf(Tee::PriDebug, "\n");

        break;
    case 1:
        if (ini_control_transfer(lvr_winusb) < 0) {
            return nullptr;
        }
        
        control_transfer(lvr_winusb, uTemperatura );
        interrupt_read(lvr_winusb);
 
        control_transfer(lvr_winusb, uIni1 );
        interrupt_read(lvr_winusb);
 
        control_transfer(lvr_winusb, uIni2 );
        interrupt_read(lvr_winusb);
        interrupt_read(lvr_winusb);
        break;
    }

    Printf(Tee::PriDebug, "pcsensor: device_type=%d\n", device_type(lvr_winusb));

    return lvr_winusb;
}

void PcSensor::pcsensor_close(usb_dev_handle* lvr_winusb){
    usb_release_interface(lvr_winusb, INTERFACE1);
    usb_release_interface(lvr_winusb, INTERFACE2);

    usb_close(lvr_winusb);
}

float PcSensor::pcsensor_get_temperature(usb_dev_handle* lvr_winusb){
    float tempc = PcSensor::ABS_ZERO;
    int ret = -1;
    switch(device_type(lvr_winusb)){
    case 0:
        ret = get_temperature(lvr_winusb, &tempc);
        break;
    case 1:
        control_transfer(lvr_winusb, uTemperatura );
        ret = interrupt_read_temperatura(lvr_winusb, &tempc);
        break;
    }
    if(ret < 0){
        return PcSensor::ABS_ZERO;
    }
    return tempc;
}
