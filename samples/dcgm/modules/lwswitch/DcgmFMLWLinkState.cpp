
#include "DcgmFMLWLinkState.h"
#include "lwlink_lib_ioctl.h"

std::string
DcgmFMLWLinkState::getMainLinkState(uint32 linkMode)
{
    switch (linkMode) {
        case lwlink_link_mode_off:
            return "Init/Off";
        case lwlink_link_mode_active:
            return "Active";
        case lwlink_link_mode_swcfg:
            return "Swcfg";
        case lwlink_link_mode_fault:
            return "Faulty";
        case lwlink_link_mode_recovery:
            return "Recovery";
        case lwlink_link_mode_fail:
            return "Fail";
        case lwlink_link_mode_detect:
            return "Detect";
        case lwlink_link_mode_reset:
            return "Reset";
        case lwlink_link_mode_enable_pm:
            return "Enable PM";
        case lwlink_link_mode_disable_pm:
            return "Disable PM";
        case lwlink_link_mode_traffic_setup:
            return "Setup Traffic";
        case lwlink_link_mode_unknown:
            return "Unknown";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}

std::string
DcgmFMLWLinkState::getTxSubLinkState(uint32 txSubLinkMode)
{
    switch (txSubLinkMode) {
        case lwlink_tx_sublink_mode_hs:
            return "High Speed";
        case lwlink_tx_sublink_mode_single_lane:
            return "Single Lane";
        case lwlink_tx_sublink_mode_train:
            return "Training";
        case lwlink_tx_sublink_mode_safe:
            return "Safe";
        case lwlink_tx_sublink_mode_off:
            return "Off";
        case lwlink_tx_sublink_mode_common_mode:
            return "Common Mode Enable";
        case lwlink_tx_sublink_mode_common_mode_disable:
            return "Common Mode Disable";
        case lwlink_tx_sublink_mode_data_ready:
            return "Data Ready";
        case lwlink_tx_sublink_mode_tx_eq:
            return "Equalization";
        case lwlink_tx_sublink_mode_pbrs_en:
            return "PRBS Generator";
        case lwlink_tx_sublink_mode_post_hs:
            return "Post Active HW Retraining";
        case lwlink_tx_sublink_mode_unknown:
            return "Unknown";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}

std::string
DcgmFMLWLinkState::getRxSubLinkState(uint32 rxSubLinkMode)
{
    switch (rxSubLinkMode) {
        case lwlink_rx_sublink_mode_hs:
            return "High Speed";
        case lwlink_rx_sublink_mode_single_lane:
            return "Single Lane";
        case lwlink_rx_sublink_mode_train:
            return "Training";
        case lwlink_rx_sublink_mode_safe:
            return "Safe";
        case lwlink_rx_sublink_mode_off:
            return "Off";
        case lwlink_rx_sublink_mode_rxcal:
            return "Calibration";
        case lwlink_rx_sublink_mode_unknown:
            return "Unknown";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}
