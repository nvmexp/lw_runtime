#ifndef TEST_DCGM_DIAG_STRINGS_H
#define TEST_DCGM_DIAG_STRINGS_H

/* Colwenience macro for embedding JSON text */
#ifndef MULTILINE
#define MULTILINE(...) #__VA_ARGS__
#endif

#include <string>

const std::string LWVS_1_3_JSON = MULTILINE(
{
    "DCGM GPU Diagnostic": {
        "test_categories": [
            {
                "category": "Deployment",
                "tests": [
                    {
                        "name": "Blacklist",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "LWML Library",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "LWCA Main Library",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Permissions and OS-related Blocks",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Persistence Mode",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Elwironmental Variables",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Page Retirement",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Graphics Processes",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Inforom",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    }
                ]
            },
            {
                "category": "Hardware",
                "tests": [
                    {
                        "name": "Memory",
                        "results": [
                            {
                                "gpu_ids": "0",
                                "status": "SKIP",
                                "warnings": [
                                    "ECC memory on this card is not lwrrently active. Please use \"lwpu-smi -i 0 -e 1\" to enable and reboot."
                                ]
                            },
                            {
                                "gpu_ids": "1",
                                "info": [
                                    "Allocated 16583361795 bytes (97.2%)"
                                ],
                                "status": "PASS"
                            },
                            {
                                "gpu_ids": "2",
                                "info": [
                                    "Allocated 16583361795 bytes (97.2%)"
                                ],
                                "status": "PASS"
                            },
                            {
                                "gpu_ids": "3",
                                "info": [
                                    "Allocated 16583361795 bytes (97.2%)"
                                ],
                                "status": "PASS"
                            }
                        ]
                    },
                    {
                        "name": "Diagnostic",
                        "results": [
                            {
                                "gpu_ids": "0,1,2,3",
                                "status": "SKIP",
                                "warnings": [
                                    "Unable to find hardware diagnostic, skipping this test."
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "category": "Integration",
                "tests": [
                    {
                        "name": "PCIe",
                        "results": [
                            {
                                "gpu_ids": "0,1,2,3",
                                "info": [  "GPU 0 GPU to Host bandwidth:\t\t13.08 GB/s",  "GPU 1 GPU to Host bandwidth:\t\t13.07 GB/s",  "GPU 2 GPU to Host bandwidth:\t\t13.07 GB/s",  "GPU 3 GPU to Host bandwidth:\t\t13.07 GB/s",  "GPU 0 Host to GPU bandwidth:\t\t12.13 GB/s",  "GPU 1 Host to GPU bandwidth:\t\t12.13 GB/s",  "GPU 2 Host to GPU bandwidth:\t\t12.13 GB/s",  "GPU 3 Host to GPU bandwidth:\t\t12.13 GB/s",  "GPU 0 bidirectional bandwidth:\t22.42 GB/s",  "GPU 1 bidirectional bandwidth:\t22.41 GB/s",  "GPU 2 bidirectional bandwidth:\t22.41 GB/s",  "GPU 3 bidirectional bandwidth:\t22.41 GB/s",  "GPU 0 GPU to Host latency:\t\t4.386 us",  "GPU 1 GPU to Host latency:\t\t3.269 us",  "GPU 2 GPU to Host latency:\t\t2.698 us",  "GPU 3 GPU to Host latency:\t\t2.349 us",  "GPU 0 Host to GPU latency:\t\t4.200 us",  "GPU 1 Host to GPU latency:\t\t3.228 us",  "GPU 2 Host to GPU latency:\t\t3.016 us",  "GPU 3 Host to GPU latency:\t\t2.996 us",  "GPU 0 bidirectional latency:\t\t7.482 us",  "GPU 1 bidirectional latency:\t\t6.525 us",  "GPU 2 bidirectional latency:\t\t5.507 us",  "GPU 3 bidirectional latency:\t\t4.785 us"
                                ],
                                "status": "PASS"
                            }
                        ]
                    }
                ]
            },
            {
                "category": "Stress",
                "tests": [
                    {
                        "name": "Memory Bandwidth",
                        "results": [
                            {
                                "gpu_ids": "0,1,2,3",
                                "status": "SKIP",
                                "warnings": [
                                    "The memory bandwidth test is skipped for this GPU."
                                ]
                            }
                        ]
                    },
                    {
                        "name": "SM Stress",
                        "results": [
                            {
                                "gpu_ids": "0,1,2,3",
                                "info": [  "GPU 0 temperature average:\t31 C",  "GPU 1 relative stress level:\t577",  "GPU 1 temperature average:\t32 C",  "GPU 2 temperature average:\t30 C",  "GPU 3 temperature average:\t33 C"
                                ],
                                "status": "FAIL",
                                "warnings": [
                                    "Max stress level of 2128.7 did not exceed desired stress level of 3000.0 for GPU 0",
                                    "Max stress level of 1096.5 did not exceed desired stress level of 3000.0 for GPU 2",
                                    "Max stress level of 1099.7 did not exceed desired stress level of 3000.0 for GPU 3"
                                ]
                            }
                        ]
                    },
                    {
                        "name": "Targeted Stress",
                        "results": [
                            {
                                "gpu_ids": "0,1,2,3",
                                "status": "SKIP",
                                "warnings": null
                            }
                        ]
                    },
                    {
                        "name": "Targeted Power",
                        "results": [
                            {
                                "gpu_ids": "0,1,2,3",
                                "status": "SKIP",
                                "warnings": null
                            }
                        ]
                    }
                ]
            }
        ],
        "version": "1.3"
    }
}
);

const std::string PER_GPU_JSON = MULTILINE(
{
    "DCGM GPU Diagnostic": {
        "test_categories": [
            {
                "category": "Deployment",
                "tests": [
                    {
                        "name": "Blacklist",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "LWML Library",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "LWCA Main Library",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Permissions and OS-related Blocks",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Persistence Mode",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Elwironmental Variables",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Page Retirement",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Graphics Processes",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    },
                    {
                        "name": "Inforom",
                        "results": [
                            { "gpu_ids": "0,1,2,3", "status": "PASS" }
                        ]
                    }
                ]
            },
            {
                "category": "Hardware",
                "tests": [
                    {
                        "name": "Memory",
                        "results": [
                            {
                                "gpu_ids": "0",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "ECC memory on GPU 0 is not lwrrently active. Please use \"lwpu-smi -i 0 -e 1\" to enable and reboot.",
                                        "error_id" : 55
                                    }

                                ]
                            },
                            {
                                "gpu_ids": "1",
                                "info": [
                                    "Allocated 16583361795 bytes (97.2%)"
                                ],
                                "status": "PASS"
                            },
                            {
                                "gpu_ids": "2",
                                "info": [
                                    "Allocated 16583361795 bytes (97.2%)"
                                ],
                                "status": "PASS"
                            },
                            {
                                "gpu_ids": "3",
                                "info": [
                                    "Allocated 16583361795 bytes (97.2%)"
                                ],
                                "status": "PASS"
                            }
                        ]
                    },
                    {
                        "name": "Diagnostic",
                        "results": [
                            {
                                "gpu_ids": "0",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "GPU 0: Unable to find hardware diagnostic, skipping this test.",
                                        "error_id" : 1
                                    }
                                ]
                            },
                            {
                                "gpu_ids": "1",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "GPU 1: Unable to find hardware diagnostic, skipping this test.",
                                        "error_id" : 1
                                    }
                                ]
                            },
                            {
                                "gpu_ids": "2",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "GPU 2: Unable to find hardware diagnostic, skipping this test.",
                                        "error_id" : 1
                                    }
                                ]
                            },
                            {
                                "gpu_ids": "3",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "GPU 3: Unable to find hardware diagnostic, skipping this test.",
                                        "error_id" : 1
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "category": "Integration",
                "tests": [
                    {
                        "name": "PCIe",
                        "results": [
                            {
                                "gpu_ids": "0",
                                "info": [
                                    "GPU 0 GPU to Host bandwidth:\t\t13.08 GB/s",
                                    "GPU 0 Host to GPU bandwidth:\t\t12.13 GB/s",
                                    "GPU 0 bidirectional bandwidth:\t22.42 GB/s",
                                    "GPU 0 GPU to Host latency:\t\t4.386 us",
                                    "GPU 0 Host to GPU latency:\t\t4.200 us",
                                    "GPU 0 bidirectional latency:\t\t7.482 us"
                                ],
                                "status": "PASS"
                            },
                            {
                                "gpu_ids": "1",
                                "info": [
                                    "GPU 1 GPU to Host bandwidth:\t\t13.07 GB/s",
                                    "GPU 1 Host to GPU bandwidth:\t\t12.13 GB/s",
                                    "GPU 1 bidirectional bandwidth:\t22.41 GB/s",
                                    "GPU 1 GPU to Host latency:\t\t3.269 us",
                                    "GPU 1 Host to GPU latency:\t\t3.228 us",
                                    "GPU 1 bidirectional latency:\t\t6.525 us"
                                ],
                                "status": "PASS"
                            },
                            {
                                "gpu_ids": "2",
                                "info": [
                                    "GPU 2 GPU to Host bandwidth:\t\t13.07 GB/s",
                                    "GPU 2 Host to GPU bandwidth:\t\t12.13 GB/s",
                                    "GPU 2 bidirectional bandwidth:\t22.41 GB/s",
                                    "GPU 2 GPU to Host latency:\t\t2.698 us",
                                    "GPU 2 Host to GPU latency:\t\t3.016 us",
                                    "GPU 2 bidirectional latency:\t\t5.507 us"
                                ],
                                "status": "PASS"
                            },
                            {
                                "gpu_ids": "3",
                                "info": [
                                    "GPU 3 GPU to Host bandwidth:\t\t13.07 GB/s",
                                    "GPU 3 Host to GPU bandwidth:\t\t12.13 GB/s",
                                    "GPU 3 bidirectional bandwidth:\t22.41 GB/s",
                                    "GPU 3 GPU to Host latency:\t\t2.349 us",
                                    "GPU 3 Host to GPU latency:\t\t2.996 us",
                                    "GPU 3 bidirectional latency:\t\t4.785 us"
                                ],
                                "status": "PASS"
                            }
                        ]
                    }
                ]
            },
            {
                "category": "Stress",
                "tests": [
                    {
                        "name": "Memory Bandwidth",
                        "results": [
                            {
                                "gpu_ids": "0",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "GPU 0: The memory bandwidth test is skipped for this GPU.",
                                        "error_id" : 48
                                    }
                                ]
                            },
                            {
                                "gpu_ids": "1",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "GPU 1: The memory bandwidth test is skipped for this GPU.",
                                        "error_id" : 48
                                    }
                                ]
                            },
                            {
                                "gpu_ids": "2",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "GPU 2: The memory bandwidth test is skipped for this GPU.",
                                        "error_id" : 48
                                    }
                                ]
                            },
                            {
                                "gpu_ids": "3",
                                "status": "SKIP",
                                "warnings": [
                                    {
                                        "warning" : "GPU 3: The memory bandwidth test is skipped for this GPU.",
                                        "error_id" : 48
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "name": "SM Stress",
                        "results": [
                            {
                                "gpu_ids": "0",
                                "info": [
                                    "GPU 0 temperature average:\t31 C"
                                ],
                                "status": "FAIL",
                                "warnings": [
                                    {
                                        "warning" : "Max stress level of 2128.7 did not exceed desired stress level of 3000.0 for GPU 0",
                                        "error_id" : 50
                                    }
                                ]
                            },
                            {
                                "gpu_ids": "1",
                                "info": [
                                    "GPU 1 relative stress level:\t577",
                                    "GPU 1 temperature average:\t32 C"
                                ],
                                "status": "PASS"
                            },
                            {
                                "gpu_ids": "2",
                                "info": [
                                    "GPU 2 temperature average:\t30 C"
                                ],
                                "status": "FAIL",
                                "warnings": [
                                    {
                                       "warning" : "Max stress level of 1096.5 did not exceed desired stress level of 3000.0 for GPU 2",
                                       "error_id" : 50
                                    }
                                ]
                            },
                            {
                                "gpu_ids": "3",
                                "info": [
                                    "GPU 3 temperature average:\t33 C"
                                ],
                                "status": "FAIL",
                                "warnings": [
                                    {
                                        "warning" : "Max stress level of 1099.7 did not exceed desired stress level of 3000.0 for GPU 3",
                                        "error_id" : 50
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "name": "Targeted Stress",
                        "results": [
                            {
                                "gpu_ids": "0",
                                "status": "SKIP"
                            },
                            {
                                "gpu_ids": "1",
                                "status": "SKIP"
                            },
                            {
                                "gpu_ids": "2",
                                "status": "SKIP"
                            },
                            {
                                "gpu_ids": "3",
                                "status": "SKIP"
                            }
                        ]
                    },
                    {
                        "name": "Targeted Power",
                        "results": [
                            {
                                "gpu_ids": "0",
                                "status": "SKIP"
                            },
                            {
                                "gpu_ids": "1",
                                "status": "SKIP"
                            },
                            {
                                "gpu_ids": "2",
                                "status": "SKIP"
                            },
                            {
                                "gpu_ids": "3",
                                "status": "SKIP"
                            }
                        ]
                    }
                ]
            }
        ],
        "version": "1.7"
    }
}
);

#endif // TEST_DCGM_DIAG_STRINGS_H

