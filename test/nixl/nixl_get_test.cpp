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

#define NUM_TRANSFERS 2
#define NUM_THREADS 1
#define SIZE 1024
//#define MEM_VAL 0xBB
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

    snprintf(now, 40, "[XX:%02ld:%02ld.%06ld] ", (ts.tv_sec%3600)/60, ts.tv_sec%60, ts.tv_nsec);

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

static std::vector<std::unique_ptr<uint8_t[]>> initMem(nixlAgent &agent,
                                                       nixl_reg_dlist_t &dram,
                                                       nixl_opt_args_t *extra_params,
                                                       uint8_t base_val, std::vector<int> sizes,
                                                       std::string role) {
    std::vector<std::unique_ptr<uint8_t[]>> addrs;

    int num = sizes.size();

    for (int i = 0; i < num; i++) {
        int size = sizes[i];
        uint8_t val = (base_val == 0) ? 0 : base_val + i;

        auto addr = std::make_unique<uint8_t[]>(size);

        std::fill_n(addr.get(), size, val);
        std::cout << "Allocating:" << (void *)addr.get() << ", "
                  << "Length:" << size << ", "
                  << "Setting to 0x" << std::hex << (unsigned)val << std::dec << std::endl;
        std::string meta = role + std::to_string(i);
        dram.addDesc(nixlBlobDesc((uintptr_t)(addr.get()), size, 0, meta));

        addrs.push_back(std::move(addr));
    }
    agent.registerMem(dram, extra_params);

    return addrs;
}

static bool checkAndDumpMem(std::vector<std::unique_ptr<uint8_t[]>> &addrs, std::vector<int> sizes, int base_val)
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



static void targetThread(nixlAgent &agent, nixl_opt_args_t *extra_params, int thread_id) {
    FUNC_IN << "args(thread_id:" << thread_id << ")\n";

    nixl_status_t ret;

    LOG << "agent.getLocalMD()\n";
    nixl_blob_t tgt_metadata;
    agent.getLocalMD(tgt_metadata);
    LOG << "LocalMD:" << tgt_metadata << std::endl;

/*
    nixl_opt_args_t md_extra_params;
    md_extra_params.ipAddr = "127.0.0.1";
    LOG << "nixlAgent.fetchRemoteMD()\n";
    agent.fetchRemoteMD(target, &md_extra_params);
    show_nixl_opt_args("md_extra_params", &md_extra_params);
    {
        std::cout << current_time() << __func__ << thread_id << "() Check Remote MD\n";
        nixl_xfer_dlist_t descs(DRAM_SEG);
        nixl_status_t check_remote;
        do {
            // spin lock
            check_remote = agent.checkRemoteMD(initiator, descs);
        } while (check_remote != NIXL_SUCCESS);
        std::cout << current_time() << __func__ << thread_id << "() Check Remote MD == DONE\n";
    }
*/

    LOG << "Get notify initiator information\n";
    nixlSerDes remote_serdes;
    {
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

        if (notifs[initiator].empty()) {
            ERR << "Get Notifs but empty.\n";
            exit(-1);
        }
        LOG << "Keep Initiator's notif\n";
        for (const auto &notif : notifs[initiator]) {
            LOG << "notif:" << notif;
            remote_serdes.importStr(notif);
        }
    }
    LOG << "So, as here, target knows initiators memory info(num and size)!\n";
    LOG << "Verify Deserialized Initiator's Desc List\n";
    nixl_xfer_dlist_t dram_initiator_ucx(&remote_serdes);
    dram_initiator_ucx.print();
    std::vector<int> sizes;
    for (int i = 0; i < dram_initiator_ucx.descCount(); i++) {
        sizes.push_back(dram_initiator_ucx[i].len);
    }

    LOG << "Now, Target Memory Allocate.\n";
    nixl_reg_dlist_t dram_for_ucx(DRAM_SEG);
    auto addrs = initMem(agent, dram_for_ucx, extra_params, 0, sizes, "target");
    dram_for_ucx.print();

    LOG << "Verify Deserialized Target's Desc List\n";
    nixl_xfer_dlist_t dram_target_ucx = dram_for_ucx.trim();
    dram_target_ucx.print();

    // Need to do this in a loop with NIXL_ERR_NOT_FOUND
    // UCX AM with desc list is faster than listener thread can recv/load MD with sockets
    // Will be deprecated with ETCD or callbacks
    extra_params->notifMsg = "NoTiFiCaTiOn";
    extra_params->hasNotif = true;
    nixlXferReqH *treq;
    ret = NIXL_SUCCESS;
    LOG << "agent.createXferReq()\n";
    do {
        ret = agent.createXferReq(NIXL_READ, dram_target_ucx, dram_initiator_ucx,
                                  initiator, treq, extra_params);
    } while (ret == NIXL_ERR_NOT_FOUND);

    if (ret != NIXL_SUCCESS) {
        ERR << "Error creating transfer request " << ret << "\n";
        exit(-1);
    }

    LOG << "Post the request with UCX backend agent.postXferReq()\n";
    ret = agent.postXferReq(treq);

    LOG << " Waiting for completion\n";
    while (ret != NIXL_SUCCESS) {
        ret = agent.getXferStatus(treq);
        assert(ret >= 0);
    }

    LOG << "Sanity Check\n";
    bool rc = checkAndDumpMem(addrs, sizes, MEM_VAL);
    if (!rc) {
        ERR << "UCX Transfer failed, buffers are different\n";
    } else {
        LOG << "Transfer completed and Buffers match with Initiator\n";
        LOG << "UCX Transfer Success!!!\n";
    }

    LOG << "Completed Sending Data using UCX backend\n";
    agent.releaseXferReq(treq);

    LOG << "Cleanup..\n";
    agent.deregisterMem(dram_for_ucx, extra_params);

    LOG << "Thead exit()\n";
}

static void runTarget(const std::string &ip, int port, nixl_thread_sync_t sync_mode) {
    FUNC_IN << "args(ip:" << ip << ", port:" << port << ", sync_mode:" << (int)sync_mode << ")\n";

    nixlAgentConfig cfg(true, true, port, sync_mode);
    cfg.pthrDelay = 1000*1000; //in us

    LOG << "Starting Agent for target\n";
    nixlAgent agent(target, cfg);

    nixl_b_params_t params = {
        //{ "num_workers", "4" },
        { "num_workers", "1" },
    };
    nixlBackendH *ucx;
    LOG << "calls nixlAgent.createBackend()\n";
    agent.createBackend("UCX", params, ucx);

    LOG << "Create extra_params\n";
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);
    show_nixl_opt_args("extra_params", &extra_params);

    LOG << "create threads\n";
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(targetThread, std::ref(agent), &extra_params, i);

    LOG << "thread will join\n";
    for (auto &thread : threads)
        thread.join();
}

static void initiatorThread(nixlAgent &agent, nixl_opt_args_t *extra_params,
                            const std::string &target_ip, int target_port, int thread_id,
                            SharedNotificationState &shared_state) {
    FUNC_IN << "args(" << target_ip << ", port:" << target_port << ", thread_id:" << thread_id << ")\n";

    LOG << "Create Initiator's memory\n";
    std::string json_string = "{\"CreateDate\":'2025/05/05 12:00:01.000', \"name\",\"sample\", \"color\":true, \"systemID\":1234}";
    int json_string_len = json_string.length();
    std::vector<int> sizes = {json_string_len, 1024, 512, 768};
    nixl_reg_dlist_t dram_for_ucx(DRAM_SEG);
    auto addrs = initMem(agent, dram_for_ucx, extra_params, MEM_VAL, sizes, "initiator");
    memcpy(addrs[0].get(), json_string.c_str() , json_string_len);
    checkAndDumpMem(addrs, sizes, MEM_VAL);

    nixl_opt_args_t md_extra_params;
    md_extra_params.ipAddr = target_ip;
    md_extra_params.port = target_port;

    LOG << "nixlAgent.fetchRemoteMD()\n";
    agent.fetchRemoteMD(target, &md_extra_params);

    LOG << "nixlAgent.sendLocalMD()\n";
    agent.sendLocalMD(&md_extra_params);

    // Notify initiator information to target
    nixlSerDes serdes;
    assert(dram_for_ucx.trim().serialize(&serdes) == NIXL_SUCCESS);
    std::string message = serdes.exportStr();

    extra_params->ipAddr = target_ip;
    extra_params->port = target_port;
    {
        LOG << "will call nixlAgent.genNotif() to tell memory info in While loop\n";
        nixl_status_t st_notif = NIXL_ERR_UNKNOWN;
        do {
            // spin lock
            LOG << "call nixlAgent.genNotif()\n";
            st_notif = agent.genNotif(target, message, extra_params);
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
    agent.invalidateLocalMD(&md_extra_params);
    show_nixl_opt_args("md_extra_params", &md_extra_params);

    LOG << "Cleanup.. call nixlAgent.deregisterMem()\n";
    agent.deregisterMem(dram_for_ucx, extra_params);
    show_nixl_opt_args("extra_params", extra_params);

    LOG << "Thead exit()\n";
}

static void runInitiator(const std::string &target_ip, int target_port, nixl_thread_sync_t sync_mode) {
    FUNC_IN << "args(" << target_ip << ", port:" << target_port << ")\n";

    nixlAgentConfig cfg(true, true, 0, sync_mode);
    cfg.pthrDelay = 1000*1000; //in us

    LOG << "Starting Agent for initiator\n";
    nixlAgent agent(initiator, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    //nixl_b_params_t params = {
    //    { "num_workers", "4" },
    //};
    nixlBackendH *ucx;

    LOG << "calls nixlAgent.getPluginParams()\n";
    agent.getPluginParams("UCX", mems, params);
    LOG << "calls nixlAgent.createBackend()\n";
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    {
        nixl_b_params_t init1;
        nixl_mem_list_t mems1;
        nixl_status_t ret1;
        LOG << "calls nixlAgent.getBackendParams()\n";
        ret1 = agent.getBackendParams(ucx, mems1, init1);
        LOG << "called nixlAgent.getBackendParams()=" << ret1 << std::endl;
        printParams(init1, mems1);
    }

    SharedNotificationState shared_state;

    LOG << "create threads\n";
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(initiatorThread, std::ref(agent), &extra_params,
                             target_ip, target_port, i, std::ref(shared_state));

    LOG "thread will join\n";
    for (auto &thread : threads)
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

    if (!role.compare(initiator) && !role.compare(target)) {
        LOG << "Invalid role. Use 'initiator' or 'target'. Currently "<< role <<std::endl;
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

    return 0;
}
