/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>
#include <chrono>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"
#include "agent_data.h"
#include <mutex>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

#define MEM_VAL 0x41

#define LOG      std::cout << current_time() << " " << gettid() << " nixl_get_test.c:" <<__LINE__ << "] "
#define ERR      std::cerr << current_time() << " " << gettid() << " nixl_get_test.c:" <<__LINE__ << "] "
#define FUNC_IN  std::cout << current_time() << " " << gettid() << "] " << __func__ << "() IN "
#define FUNC_OUT std::cout << current_time() << " " << gettid() << "] " << __func__ << "() OUT "

void printParams(const nixl_b_params_t& params, const nixl_mem_list_t& mems) {
    if (params.empty()) {
        std::cout << "Parameters: (empty)\n";
        return;
    }

    std::cout << "Parameters:\n";
    for (const auto& pair : params) {
        std::cout << "  " << pair.first << " = " << pair.second << std::endl;
    }

    if (mems.empty()) {
        std::cout << "Mems: (empty)\n";
        return;
    }

    std::cout << "Mems:\n";
    for (const auto& elm : mems) {
        std::cout << "  " << nixlEnumStrings::memTypeStr(elm) << std::endl;
    }
}

static char* current_time(void)
{
    static char now[40];
    struct timespec ts = (struct timespec){0};
    clock_gettime(CLOCK_REALTIME, &ts);

    snprintf(now, 40, "XX:%02ld:%02ld.%06ld ", (ts.tv_sec%3600)/60, ts.tv_sec%60, ts.tv_nsec);

    return now;
}

static void show_nixl_opt_args(const std::string name, nixl_opt_args_t* params)
{
    std::cout << current_time() << ": Show nixl_opt_args_t(" << name << ")\n";
    for (nixlBackendH* b : params->backends) {
        std::cout << "  params.backends.backendType:" << b->getType() << "\n";
        std::cout << "  params.backends.supportsRemote:" << b->supportsRemote() << "\n";
        std::cout << "  params.backends.supportsLocal:" << b->supportsLocal() << "\n";
        std::cout << "  params.backends.supportsNotif:" << b->supportsNotif() << "\n";
        std::cout << "  params.backends.supportsProgTh:" << b->supportsProgTh() << "\n";
    }
    std::cout << "  params.notifMsg:" << params->notifMsg << "\n";
    std::cout << "  params.hasNotif:" << params->hasNotif << "\n";
    std::cout << "  params.skipDescMerge:" << params->skipDescMerge << "\n";
    std::cout << "  params.includeConnInfo:" << params->includeConnInfo << "\n";
    std::cout << "  params.ipAddr:" << params->ipAddr << "\n";
    std::cout << "  params.port:" << params->port << "\n";
    std::cout << "  params.metadataLabel:" << params->metadataLabel << "\n";
}

static void show_nixl_notifs_t(nixl_notifs_t &notifs)
{
    if (notifs.size()) {
        std::cout << current_time() << ": Show nixl_notifs_t\n";
        std::for_each(std::begin(notifs), std::end(notifs),
                      [](nixl_notifs_t::value_type p) {
                          //std::cout << "{" << p.first << "," << p.second << "}, ";
                          std::cout << "  {" << p.first << ",{";
                          for(auto s :p.second) { std::cout << s << ","; }
                          std::cout << "}}\n";
                      });
    }
}

/**
 * This test does p2p from using PUT.
 * intitator -> target so the metadata and
 * desc list needs to move from
 * target to initiator
 */

struct SharedNotificationState {
    std::mutex mtx;
    std::vector<nixlSerDes> remote_serdes;
};

static const std::string target("target");
static const std::string initiator("initiator");

static const std::string json_string0 = "{\"agent-num\":0, \"CreateDate\":'2025/05/05 12:00:00.000', \"name\",\"sample\", \"color\":true, \"systemID\":1234, \"comment\":\"this is for initiator0.\"}";
static int json_string0_len = json_string0.length();

static const std::string json_string1 = "{\"agent-num\":1, \"CreateDate\":'2025/05/05 12:00:01.000', \"name\",\"sample\", \"color\":true, \"systemID\":4321, \"comment\":\"Happy birthday to you~ Happy birthday to you~\"}";
static const int json_string1_len = json_string1.length();

static const std::vector<std::string> json_strings = {json_string0, json_string1};
static const std::vector<std::vector<int>> json_sizes = {{json_string0_len, 1024,  512, 768},
                                                         {json_string1_len,  384, 2048, 4*1024*1024}};

struct MunmapDeleter {
    void operator()(void* ptr) const {
        if (ptr != nullptr && size_ > 0) {
            if (msync(ptr, size_, 0) == -1) {
                ERR << "msync failed" << std::endl;
            }

            close(fd_);

            if (munmap(ptr, size_) == -1) {
                ERR << "munmap failed" << std::endl;
            }
        }
    }
    int fd_;
    size_t size_;
};

static std::vector<std::unique_ptr<uint8_t[], MunmapDeleter>> initMem(nixlAgent &agent,
                                                                      nixl_reg_dlist_t &dram,
                                                                      nixl_opt_args_t *extra_params,
                                                                      uint8_t base_val, std::vector<int> sizes,
                                                                      std::string role) {
    std::vector<std::unique_ptr<uint8_t[], MunmapDeleter>> addrs;

    int num = sizes.size();

    for (int i = 0; i < num; i++) {
        int size = sizes[i];
        uint8_t val = 0;
        if (i != 0 && base_val)
            val = base_val + i;

        auto map = malloc(size);
        std::unique_ptr<uint8_t[], MunmapDeleter> addr(static_cast<uint8_t*>(map), {MunmapDeleter{0,0}});

        std::fill_n(addr.get(), size, val);
        std::string meta = "DRAM:" + role + std::to_string(i);
        std::cout << "Allocating:" << (void *)addr.get() << ", "
                  << "Length:" << size << ", "
                  << "Setting to 0x" << std::hex << (unsigned)val << std::dec << ", "
                  << "meta info:" << meta << std::endl;
        dram.addDesc(nixlBlobDesc((uintptr_t)(addr.get()), size, 0, meta));

        addrs.push_back(std::move(addr));
    }
    agent.registerMem(dram, extra_params);

    return addrs;
}


static std::vector<std::unique_ptr<uint8_t[], MunmapDeleter>> initFile(nixlAgent &agent,
                                                                       nixl_reg_dlist_t &dram,
                                                                       nixl_reg_dlist_t &file,
                                                                       nixl_opt_args_t *extra_params,
                                                                       uint8_t base_val, std::vector<int> sizes,
                                                                       std::string role) {
    std::vector<std::unique_ptr<uint8_t[], MunmapDeleter>> addrs;

    int num = sizes.size();

    {
        int size = sizes[0];
        uint8_t val = 0;

        auto map = malloc(size);
        std::unique_ptr<uint8_t[], MunmapDeleter> addr(static_cast<uint8_t*>(map), {MunmapDeleter{0,0}});

        std::fill_n(addr.get(), size, val);
        std::string meta = "DRAM:" + role + std::to_string(0);
        std::cout << "Allocating:" << (void *)addr.get() << ", "
                  << "Length:" << size << ", "
                  << "Setting to 0x0, "
                  << "meta info:" << meta << std::endl;
        dram.addDesc(nixlBlobDesc((uintptr_t)(addr.get()), size, 0, meta));
        addrs.push_back(std::move(addr));
        agent.registerMem(dram, extra_params);
    }

    for (int i = 1; i < num; i++) {
        size_t size = sizes[i];
        std::string meta = "FILE:" + role + std::to_string(i);
        std::string file_name = "file_" + role + std::to_string(i) + ".txt";

        int fd = open(file_name.c_str(), O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);
        if (fd < 0) {
            ERR << "Can not create " << file_name << std::endl;
            exit(-1);
        }
        // create empty file
        if(ftruncate(fd, size)){
            ERR << "Fail ftruncate()\n";
            exit(-1);
        }

        auto map = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            ERR << "mmap failed" << std::endl;
            close(fd);
            exit(-1);
        }
        std::unique_ptr<uint8_t[], MunmapDeleter> addr(static_cast<uint8_t*>(map), {MunmapDeleter{fd,size}});

        std::cout << "Allocating mapped file:" << (void *)addr.get() << ", "
                  << "Length:" << size << ", "
                  << "Mapped Length:" << size << ", "
                  << "meta info:" << meta << std::endl;
        file.addDesc(nixlBlobDesc((uintptr_t)(addr.get()), size, 0, meta));

        addrs.push_back(std::move(addr));
    }
    agent.registerMem(file, extra_params);

    return addrs;
}

static bool checkAndDumpMem(std::vector<std::unique_ptr<uint8_t[], MunmapDeleter>> &addrs, std::vector<int> sizes, int base_val)
{
    if (addrs.size() != sizes.size())
        return false;

    bool result = true;
    int num = sizes.size();
    std::string a((char*)addrs[0].get());
    LOG << "Check and Dump memoryes:\n";
    std::cout << "  [0]:" << a << std::endl;

    for (int i = 1; i < num; i++) {
        int len = sizes[i];
        auto addr = addrs[i].get();
        uint8_t val = (base_val == 0) ? 0 : base_val + i;
        std::cout << "  [" << i << "](len:" << len << "):";
        for (int j = 0; j < len; j++) {
            if (j < 20) {
                printf("0x%X ", addr[j]);
            }
            if (addr[j] != val)
                result = false;
        }
        std::cout << std::endl;
    }
    return result;
}

static void targetThread(nixlAgent &agent, nixl_opt_args_t *extra_params) {
    FUNC_IN << "()\n";

    nixl_status_t ret;

    LOG << "agent.getLocalMD()\n";
    nixl_blob_t tgt_metadata;
    agent.getLocalMD(tgt_metadata);
    LOG << "LocalMD:" << tgt_metadata << std::endl;

    LOG << "Get notify remote information\n";
    nixlSerDes remote_serdes;
    std::string remote_agent_name;

    LOG << "will call nixlAgent.getNotif() in While loop\n";
    nixl_notifs_t notifs;
    do {
        // spin lock
        //LOG << "nixlAgent.getNotifs()\n";
        ret = agent.getNotifs(notifs, extra_params);
        if (notifs.size() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        //LOG << "nixlAgent.getNotifs()=" << ret << ", notifs.size()=" << notifs.size() << std::endl;
    } while(ret != NIXL_SUCCESS || notifs.size() == 0);
    LOG << "will call nixlAgent.getNotif() DONE\n";

    show_nixl_notifs_t(notifs);
    for (auto iter = notifs.begin(); iter != notifs.end(); iter++) {
        LOG << "Remote Agent Name: " << iter->first << "\n";
        remote_agent_name = iter->first;

        if (notifs[remote_agent_name].empty()) {
            ERR << "Get Notifs but empty.\n";
            exit(-1);
        }
        LOG << "Keep Remote's notif\n";
        for (const auto &notif : notifs[remote_agent_name]) {
            LOG << "notif:" << notif << std::endl;
            remote_serdes.importStr(notif);
        }

        LOG << "So, as here, target knows remotes memory info(num and size)!\n";
        LOG << "Verify Deserialized Remote's Desc List\n";
        nixl_reg_dlist_t dram_remote_ucx(&remote_serdes);
        dram_remote_ucx.print();
        std::vector<int> sizes;
        for (int i = 0; i < dram_remote_ucx.descCount(); i++) {
            sizes.push_back(dram_remote_ucx[i].len);
        }

        LOG << "Now, Target Memory Allocate.\n";
        nixl_reg_dlist_t dram_for_ucx(DRAM_SEG);
        nixl_reg_dlist_t file_for_ucx(FILE_SEG);
        auto addrs = initFile(agent, dram_for_ucx, file_for_ucx, extra_params, 0, sizes, "target");
        LOG << "dram_for_ucx;\n";
        dram_for_ucx.print();
        LOG << "file_for_ucx;\n";
        file_for_ucx.print();

        LOG << "Verify Deserialized Target's DRAM Desc List\n";
        nixl_xfer_dlist_t xfer_target_dram_ucx = dram_for_ucx.trim();
        xfer_target_dram_ucx.print();
        LOG << "Verify Deserialized Target's FILE Desc List\n";
        nixl_xfer_dlist_t xfer_target_file_ucx = file_for_ucx.trim();
        xfer_target_file_ucx.print();

        LOG << "Verify Deserialized Remote's DRAM Desc List\n";
        nixl_xfer_dlist_t xfer_remote_dram_ucx(DRAM_SEG);
        xfer_remote_dram_ucx.addDesc(dram_remote_ucx[0]);
        xfer_remote_dram_ucx.print();
        LOG << "Verify Deserialized Remote's FILE Desc List\n";
        nixl_xfer_dlist_t xfer_remote_file_ucx(DRAM_SEG);
        xfer_remote_file_ucx.addDesc(dram_remote_ucx[1]);
        xfer_remote_file_ucx.addDesc(dram_remote_ucx[2]);
        xfer_remote_file_ucx.addDesc(dram_remote_ucx[3]);
        xfer_remote_file_ucx.print();

        nixlXferReqH *treq;
        //nixlXferReqH *treq2;
        {
            LOG << "agent.createXferReq() for DRAM\n";
            do {
                ret = agent.createXferReq(NIXL_READ, xfer_target_dram_ucx, xfer_remote_dram_ucx,
                                          remote_agent_name, treq, extra_params);
            } while (ret == NIXL_ERR_NOT_FOUND);
            if (ret != NIXL_SUCCESS) {
                ERR << "Error creating transfer request " << ret << "\n";
                exit(-1);
            }

            LOG << "Post the request with UCX backend agent.postXferReq() for DRAM\n";
            ret = agent.postXferReq(treq);

            LOG << " Waiting for completion for DRAM\n";
            while (ret != NIXL_SUCCESS) {
                ret = agent.getXferStatus(treq);
                assert(ret >= 0);
            }

            LOG << "Completed Sending Data using UCX backend for DRAM\n";
            agent.releaseXferReq(treq);
            treq = nullptr;
        }
        {
            LOG << "agent.createXferReq() for FILE\n";
            do {
                ret = agent.createXferReq(NIXL_READ, xfer_target_file_ucx, xfer_remote_file_ucx,
                                          remote_agent_name, treq, extra_params);
            } while (ret == NIXL_ERR_NOT_FOUND);
            if (ret != NIXL_SUCCESS) {
                ERR << "Error creating transfer request " << ret << "\n";
                exit(-1);
            }

            LOG << "Post the request with UCX backend agent.postXferReq() for FILE\n";
            ret = agent.postXferReq(treq);

            LOG << " Waiting for completion for FILE\n";
            while (ret != NIXL_SUCCESS) {
                ret = agent.getXferStatus(treq);
                assert(ret >= 0);
            }

            LOG << "Completed Sending Data using UCX backend for FILE\n";
            agent.releaseXferReq(treq);
        }
        {
            // send end message
            nixl_status_t st_notif;
            std::string message = "NoTiFiCaTiOn";
            do {
                // spin lock
                //LOG << "call nixlAgent.genNotif()\n";
                st_notif = agent.genNotif(remote_agent_name, message, extra_params);
                if (st_notif == NIXL_ERR_NOT_FOUND) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                //LOG << "call nixlAgent.genNotif()=" << st_notif << std::endl;
            } while (st_notif != NIXL_SUCCESS);
            LOG << "CALLED nixlAgent.genNotif()\n";
        }

        LOG << "Sanity Check\n";
        bool rc = checkAndDumpMem(addrs, sizes, MEM_VAL);
        if (!rc) {
            ERR << "UCX Transfer failed, buffers are different\n";
        } else {
            LOG << "Transfer completed and Buffers match with Remote\n";
            LOG << "UCX Transfer Success!!!\n";
        }


        LOG << "Cleanup memory DRAM\n";
        agent.deregisterMem(dram_for_ucx, extra_params);
        LOG << "Cleanup memory FILE\n";
        agent.deregisterMem(file_for_ucx, extra_params);

        LOG << "Cleanup remote agent. agent.invalidateRemoteMD()\nThis must be needed for next transfer.";
        agent.invalidateRemoteMD(remote_agent_name);
    } //for (auto iter = notifs.begin();

    LOG << "Thead exit()\n";
}

static void runTarget(const std::string &ip, int port, nixl_thread_sync_t sync_mode) {
    FUNC_IN << "args(ip:" << ip << ", port:" << port << ", sync_mode:" << (int)sync_mode << ")\n";

    // delete files
    system("rm -f file_*.txt");

    nixlAgentConfig cfg(true, true, port, sync_mode);
    cfg.pthrDelay = 1000*1000; //in us

    LOG << "Starting Agent for target\n";
    nixlAgent agent(target, cfg);

    LOG << "calls nixlAgent.createBackend(UCX)\n";
    nixl_b_params_t params = {
        { "num_workers", "1" },
    };
    nixlBackendH *ucx;
    agent.createBackend("UCX", params, ucx);

    LOG << "Create extra_params for UCX\n";
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);
    show_nixl_opt_args("extra_params", &extra_params);

    LOG << "create threads\n";
    std::thread thread(targetThread, std::ref(agent), &extra_params);

    LOG << "thread will join\n";
    thread.join();
}

static void initiatorThread(nixlAgent &agent, nixl_opt_args_t *extra_params, nixl_opt_args_t* md_extra_params) {
    FUNC_IN << "()\n";

    LOG << "nixlAgent.fetchRemoteMD()\n";
    agent.fetchRemoteMD(target, md_extra_params);
    // fetch された remoteMD は非同期で受信され、受信されないと genNotif が成功しない
    // 一回受信できたら、その後は genNotif はすぐ成功する

    SharedNotificationState shared_state;

    LOG << "Create Initiator's memory\n";
    nixl_reg_dlist_t dram_for_ucx(DRAM_SEG);
    const int thread_id = 0;
    auto addrs = initMem(agent, dram_for_ucx, extra_params, MEM_VAL+thread_id, json_sizes[thread_id], initiator);
    memcpy(addrs[0].get(), json_strings[thread_id].c_str() , strlen(json_strings[thread_id].c_str()));
    checkAndDumpMem(addrs, json_sizes[thread_id], MEM_VAL);
    dram_for_ucx.print();

    LOG << "nixlAgent.sendLocalMD()\n";
    agent.sendLocalMD(md_extra_params);

    // Notify initiator information to target
    nixlSerDes serdes;
    assert(dram_for_ucx.serialize(&serdes) == NIXL_SUCCESS);
    std::string message = serdes.exportStr();

    {
        LOG << "will call nixlAgent.genNotif() to tell memory info in While loop\n";
        nixl_status_t st_notif;
        do {
            // spin lock
            //LOG << "call nixlAgent.genNotif()\n";
            st_notif = agent.genNotif(target, message, extra_params);
            if (st_notif == NIXL_ERR_NOT_FOUND) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            //LOG << "call nixlAgent.genNotif()=" << st_notif << std::endl;
        } while (st_notif != NIXL_SUCCESS);
        LOG << "CALLED nixlAgent.genNotif()\n";
    }

    {
        LOG << "wait NotifAmCallback\n";
        nixl_status_t st_getNotif;
        nixl_notifs_t am_notifs;
        do {
            // spin lock
            //LOG << "call nixlAgent.getNotif()\n";
            st_getNotif = agent.getNotifs(am_notifs, extra_params);
            if (am_notifs.size() == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } while (st_getNotif != NIXL_SUCCESS || am_notifs.size() == 0);
        LOG << "CALLED nixlAgent.getNotif()\n";
        show_nixl_notifs_t(am_notifs);
    }

    LOG << "Data is transrated.\n";

    LOG << "call nixlAgent.invalidateLocalMD()\n";
    agent.invalidateLocalMD(md_extra_params);
    show_nixl_opt_args("md_extra_params", md_extra_params);

    LOG << "Cleanup.. call nixlAgent.deregisterMem()\n";
    agent.deregisterMem(dram_for_ucx, extra_params);
    show_nixl_opt_args("extra_params", extra_params);

    LOG << "Thead exit()\n";
}

static void runInitiator(const std::string &target_ip, int target_port, nixl_thread_sync_t sync_mode) {
    FUNC_IN << "args(" << target_ip << ", port:" << target_port << ", sync_mode:" << (int)sync_mode << ")\n";

    nixlAgentConfig cfg(true, true, 0, sync_mode);
    cfg.pthrDelay = 1000*1000; //in us

    LOG << "Starting Agent for initiator\n";
    nixlAgent agent(initiator, cfg);

    LOG << "calls nixlAgent.createBackend(UCX)\n";
    nixl_b_params_t params = {
        { "num_workers", "1" },
    };
    nixlBackendH *ucx;
    agent.createBackend("UCX", params, ucx);
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);
    extra_params.ipAddr = target_ip;
    extra_params.port = target_port;
    show_nixl_opt_args("extra_params", &extra_params);

    nixl_opt_args_t md_extra_params;
    md_extra_params.ipAddr = target_ip;
    md_extra_params.port = target_port;

    LOG << "create threads\n";
    std::thread thread(initiatorThread, std::ref(agent), &extra_params, &md_extra_params);

    LOG "thread will join\n";
    thread.join();
}

int main(int argc, char *argv[]) {
    FUNC_IN << std::endl;

    /** Argument Parsing */
    if (argc < 4) {
        std::cout <<"Enter the required arguments\n";
        std::cout <<"  <Role:target,initiator> " <<"<Target IP> <Target Port>\n";
        exit(-1);
    }

    std::string role = std::string(argv[1]);
    const char  *target_ip   = argv[2];
    int         target_port = std::stoi(argv[3]);

    std::transform(role.begin(), role.end(), role.begin(), ::tolower);

    if (!role.find(initiator) && !role.compare(target)) {
        LOG << "Invalid role. Use 'initiator*' or 'target'. Currently "<< role <<std::endl;
        return 1;
    }

    auto sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    if (argc == 5) {
        std::string sync_mode_str{argv[4]};
        std::transform(sync_mode_str.begin(), sync_mode_str.end(), sync_mode_str.begin(), ::tolower);
        if (sync_mode_str == "rw") {
            sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
            LOG << "Using RW sync mode\n";
        } else if (sync_mode_str == "strict") {
            sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            LOG << "Using Strict sync mode\n";
        } else {
            ERR << "Invalid sync mode. Use 'rw' or 'strict'.\n";
            return 1;
        }
    }

    /*** End - Argument Parsing */

    if (role == target)
        runTarget(target_ip, target_port, sync_mode);
    else
        runInitiator(target_ip, target_port, sync_mode);

    FUNC_OUT << " for role:" << role << std::endl;
    return 0;
}
