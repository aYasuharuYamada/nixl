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

#define NUM_TRANSFERS 2
#define NUM_THREADS 1
#define SIZE 1024
//#define MEM_VAL 0xBB
#define MEM_VAL 0x41

#if 0
template <class C>
void print(const C& c)
{
  std::for_each(std::begin(c), std::end(c), [](typename C::value_type p) { std::cout << "{" << p.first << "," << p.second << "}, "; });
  std::cout << std::endl;
}
#endif

void printParams(const nixl_b_params_t& params, const nixl_mem_list_t& mems) {
    if (params.empty()) {
        std::cout << "Parameters: (empty)" << std::endl;
        return;
    }

    std::cout << "Parameters:" << std::endl;
    for (const auto& pair : params) {
        std::cout << "  " << pair.first << " = " << pair.second << std::endl;
    }

    if (mems.empty()) {
        std::cout << "Mems: (empty)" << std::endl;
        return;
    }

    std::cout << "Mems:" << std::endl;
    for (const auto& elm : mems) {
        std::cout << "  " << nixlEnumStrings::memTypeStr(elm) << std::endl;
    }
}

static char* current_time(void)
{
    static char now[20];
    struct timespec ts = (struct timespec){0};
    clock_gettime(CLOCK_REALTIME, &ts);

    snprintf(now, 20, "[%3ld.%09ld] ", ts.tv_sec%1000, ts.tv_nsec);

    return now;
}

#if 1
static void dump_data(const char* name, void* data, size_t len)
{
    unsigned char* c = (unsigned char*)data;
    len = (len < 100) ? len : 100;
    printf("[xxx.xxxxxxxxx] %s:", name);
    for (size_t i = 0; i < len; i++) {
      printf("0x%X ", c[i]);
    }
    printf("\n");
}
#endif

static void show_nixl_opt_args(const std::string name, nixl_opt_args_t* params)
{
    std::cout << "[xxx.xxxxxxxxx] : Show nixl_opt_args_t(" << name << ")\n";
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
    // using nixl_notifs_t = std::unordered_map<std::string, std::vector<nixl_blob_t>>;
    if (notifs.size()) {
        printf("[xxx.xxxxxxxxx] : Show nixl_notifs_t\n");
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
                                                       uint8_t val, std::string role) {
    std::vector<std::unique_ptr<uint8_t[]>> addrs;

    for (int i = 0; i < NUM_TRANSFERS; i++) {
        auto addr = std::make_unique<uint8_t[]>(SIZE);

        uint8_t _val = (val == 0) ? 0 : val + i;
        std::fill_n(addr.get(), SIZE, _val);
        std::cout << "Allocating : " << (void *)addr.get() << ", "
                  << "Setting to 0x" << std::hex << (unsigned)_val << std::dec << std::endl;
        std::string meta = role + std::to_string(i);
        dram.addDesc(nixlBlobDesc((uintptr_t)(addr.get()), SIZE, 0, meta));

        addrs.push_back(std::move(addr));
    }
    agent.registerMem(dram, extra_params);

    return addrs;
}

static void targetThread(nixlAgent &agent, nixl_opt_args_t *extra_params, int thread_id) {
    std::cout << current_time() << __func__ << thread_id << "(" << "thread_id:" << thread_id << ")" << std::endl;

    // Get notify initiator information
    {
        std::cout << current_time() << __func__ << thread_id << "() will call nixlAgent.getNotif() in While loop\n";
        //nixl_status_t ret = NIXL_ERR_UNKNOWN;
        nixl_notifs_t notifs;
        do {
            // spin lock
            //std::cout << current_time() << __func__ << thread_id << "() nixlAgent.getNotifs()" << std::endl;
            (void)agent.getNotifs(notifs, extra_params);
            //std::cout << current_time() << __func__ << thread_id << "() nixlAgent.getNotifs()=" << ret << ", notifs.size()=" << notifs.size() << std::endl;
        } while(notifs.size() == 0);

        std::cout << current_time() << __func__ << thread_id << "() will call nixlAgent.getNotif() DONE\n";
        show_nixl_notifs_t(notifs);
    }

    nixl_reg_dlist_t dram_for_ucx(DRAM_SEG);
    auto addrs = initMem(agent, dram_for_ucx, extra_params, 0, "target");

    {
        nixl_blob_t tgt_metadata;
        agent.getLocalMD(tgt_metadata);
        std::cout << current_time() << __func__ << thread_id << "() LocalMD:" << tgt_metadata  << "\n";
    }

    std::cout << current_time() << __func__ << thread_id << "() Start Control Path metadata exchanges\n";

    std::cout << current_time() << __func__ << thread_id << "() Desc List from Target to Initiator\n";
    dram_for_ucx.print();

    /** Only send desc list */
    nixlSerDes serdes;
    assert(dram_for_ucx.trim().serialize(&serdes) == NIXL_SUCCESS);

    std::cout << current_time() << __func__ << thread_id << "() Wait for initiator and then send xfer descs\n";
    std::string message = serdes.exportStr();
    std::cout << current_time() << __func__ << thread_id << "() serdes.exportStr() is " << message << "\n";
    show_nixl_opt_args("extra_params", extra_params);

    // Notify target information to initiator
    {
        std::cout << current_time() << __func__ << thread_id << "() will call nixlAgent.genNotif() in While loop\n";
        nixl_status_t st_notif = NIXL_ERR_UNKNOWN;
        do {
            // spin lock
            //std::cout << current_time() << __func__ << thread_id << "() call nixlAgent.genNotif()\n";
            st_notif = agent.genNotif(initiator, message, extra_params);
        } while (st_notif != NIXL_SUCCESS);
        std::cout << current_time() << __func__ << thread_id << "() CALLED nixlAgent.genNotif()\n";
    }
    std::cout << current_time() << __func__ << thread_id << "() End Control Path metadata exchanges\n";
    show_nixl_opt_args("extra_params", extra_params);

    std::cout << current_time() << __func__ << thread_id << "() Start Data Path Exchanges\n";
    std::cout << current_time() << __func__ << thread_id << "() Waiting to receive Data from Initiator\n";

    bool rc = false;
    int n_tries;
    for (n_tries = 0; !rc ; n_tries++) {
        //Only works with progress thread now, as backend is protected
        /** Sanity Check */
        rc = std::all_of(addrs.begin(), addrs.end(), [](auto &addr) {
            return std::all_of(addr.get(), addr.get() + SIZE, [](int x) {
                if (x != 0) {
                    return true;
                }
                return false;
            });
        });
        if (!rc) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } else {
            std::cout << current_time() << __func__ << thread_id << "() Data received after " << n_tries << " tryes\n";
            std::all_of(addrs.begin(), addrs.end(), [](auto &addr) { dump_data("", (void*)addr.get(), SIZE); return true; });
        }
    }
    if (!rc)
        std::cout << current_time() << __func__ << thread_id << "() UCX Transfer failed, buffers are different\n";
    else {
        std::cout << current_time() << __func__ << thread_id << "() Transfer completed and Buffers match with Initiator\n";
        std::cout << current_time() << __func__ << thread_id << "() UCX Transfer Success!!!\n";
    }
    std::cout << current_time() << __func__ << thread_id << "() Cleanup..  call nixlAgent.deregisterMem\n";
    agent.deregisterMem(dram_for_ucx, extra_params);

    std::cout << current_time() << __func__ << thread_id << "() Thead exit()\n";
}

static void runTarget(const std::string &ip, int port, nixl_thread_sync_t sync_mode) {
    std::cout << current_time() << __func__ << "()" << std::endl;

    nixlAgentConfig cfg(true, true, port, sync_mode);
    cfg.pthrDelay = 1000*1000; //in us

    std::cout << "Starting Agent for target\n";
    nixlAgent agent(target, cfg);

    nixl_b_params_t params = {
        //{ "num_workers", "4" },
        { "num_workers", "1" },
    };
    nixlBackendH *ucx;
    std::cout << current_time() << __func__ << "() calls nixlAgent.createBackend()" << std::endl;
    agent.createBackend("UCX", params, ucx);

    std::cout << current_time() << __func__ << "() Create extra_params" << std::endl;
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);
    show_nixl_opt_args("extra_params", &extra_params);

    std::cout << current_time() << __func__ << "() create threads" << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(targetThread, std::ref(agent), &extra_params, i);

    std::cout << current_time() << __func__ << "() thread will join" << std::endl;
    for (auto &thread : threads)
        thread.join();
}

static void initiatorThread(nixlAgent &agent, nixl_opt_args_t *extra_params,
                            const std::string &target_ip, int target_port, int thread_id,
                            SharedNotificationState &shared_state) {
    std::cout << current_time() << __func__ << thread_id << "(" << target_ip << ", port:" << target_port << ", thread_id:" << thread_id << ")" << std::endl;

    show_nixl_opt_args("extra_params", extra_params);
    nixl_reg_dlist_t dram_for_ucx(DRAM_SEG);
    auto addrs = initMem(agent, dram_for_ucx, extra_params, MEM_VAL, "initiator");
    show_nixl_opt_args("extra_params", extra_params);

    std::cout << current_time() << __func__ << thread_id << "() Start Control Path metadata exchanges\n";
    std::cout << current_time() << __func__ << thread_id << "() Exchange metadata with Target\n";

    nixl_opt_args_t md_extra_params;
    md_extra_params.ipAddr = target_ip;
    md_extra_params.port = target_port;

    std::cout << current_time() << __func__ << thread_id << "() nixlAgent.fetchRemoteMD()" << std::endl;
    agent.fetchRemoteMD(target, &md_extra_params);
    show_nixl_opt_args("md_extra_params", &md_extra_params);
    {
      std::cout << current_time() << __func__ << thread_id << "() Check Remote MD" << std::endl;
      nixl_xfer_dlist_t descs(DRAM_SEG);
      nixl_status_t check_remote;
      do {
        // spin lock
        check_remote = agent.checkRemoteMD(target, descs);
      } while (check_remote != NIXL_SUCCESS);
      std::cout << current_time() << __func__ << thread_id << "() Check Remote MD == DONE" << std::endl;
    }

    std::cout << current_time() << __func__ << thread_id << "() nixlAgent.sendLocalMD()" << std::endl;
    agent.sendLocalMD(&md_extra_params);
    show_nixl_opt_args("md_extra_params", &md_extra_params);

    // Notify initiator information to target
    {
        std::cout << current_time() << __func__ << thread_id << "() will call nixlAgent.genNotif() in While loop\n";
        nixl_status_t st_notif = NIXL_ERR_UNKNOWN;
        do {
            // spin lock
            std::cout << current_time() << __func__ << thread_id << "() call nixlAgent.genNotif()\n";
            st_notif = agent.genNotif(target, "From initiator JSON {'name':'initiator}", extra_params);
        } while (st_notif != NIXL_SUCCESS);
        std::cout << current_time() << __func__ << thread_id << "() CALLED nixlAgent.genNotif()\n";
    }

    // Wait for notifications and populate shared state
    while (true) {
        {
            std::lock_guard<std::mutex> lock(shared_state.mtx);
            if (shared_state.remote_serdes.size() >= NUM_THREADS) {
                break;
            }
        }

        nixl_notifs_t notifs;
        std::cout << current_time() << __func__ << thread_id << "() nixlAgent.getNotifs()" << std::endl;
        nixl_status_t ret = agent.getNotifs(notifs, extra_params);
        std::cout << current_time() << __func__ << thread_id << "() nixlAgent.getNotifs()=" << ret << ", notifs.size()=" << notifs.size() << std::endl;
        assert(ret >= 0);

        show_nixl_notifs_t(notifs);
        if (notifs.size() > 0) {
            std::lock_guard<std::mutex> lock(shared_state.mtx);
            for (const auto &notif : notifs[target]) {
                nixlSerDes serdes;
                serdes.importStr(notif);
                shared_state.remote_serdes.push_back(serdes);
            }
        }
    }

    // Get our thread's serdes instance
    nixlSerDes remote_serdes;
    {
        std::lock_guard<std::mutex> lock(shared_state.mtx);
        remote_serdes = shared_state.remote_serdes[thread_id];
    }

    std::cout << current_time() << __func__ << thread_id << "() Verify Deserialized Target's Desc List at Initiator\n";
    nixl_xfer_dlist_t dram_target_ucx(&remote_serdes);
    std::cout << current_time() << __func__ << thread_id << "():" << __LINE__ << "\n";
    nixl_xfer_dlist_t dram_initiator_ucx = dram_for_ucx.trim();
    std::cout << current_time() << __func__ << thread_id << "():" << __LINE__ << "\n";
    dram_target_ucx.print();

    std::cout << current_time() << __func__ << thread_id << "() End Control Path metadata exchanges\n";
    std::cout << current_time() << __func__ << thread_id << "() Start Data Path Exchanges\n\n";
    std::cout << current_time() << __func__ << thread_id << "() Create transfer request with UCX backend\n";

    extra_params->notifMsg = "NoTiFiCaTiOn";
    extra_params->hasNotif = true;
    show_nixl_opt_args("extra_params", extra_params);
    // Need to do this in a loop with NIXL_ERR_NOT_FOUND
    // UCX AM with desc list is faster than listener thread can recv/load MD with sockets
    // Will be deprecated with ETCD or callbacks
    nixlXferReqH *treq;
    nixl_status_t ret = NIXL_SUCCESS;
    do {
        std::cout << current_time() << __func__ << thread_id << "() nixlAgent.createXferReq()" << std::endl;
        ret = agent.createXferReq(NIXL_WRITE, dram_initiator_ucx, dram_target_ucx,
                                  target, treq, extra_params);
    } while (ret == NIXL_ERR_NOT_FOUND);
    std::cout << current_time() << __func__ << thread_id << "() nixlAgent.createXferReq()=" << ret << std::endl;

    if (ret != NIXL_SUCCESS) {
        std::cerr << "Thread " << thread_id << " Error creating transfer request " << ret << "\n";
        exit(-1);
    }

    show_nixl_opt_args("extra_params", extra_params);
    std::cout << current_time() << __func__ << thread_id << "() Post the request with UCX backend\n";
    std::cout << current_time() << __func__ << thread_id << "() nixlAgent.postXferReq()\n";
    ret = agent.postXferReq(treq);
    std::cout << current_time() << __func__ << thread_id << "() Initiator posted Data Path transfer\n";
    std::cout << current_time() << __func__ << thread_id << "() Waiting for completion\n";

    while (ret != NIXL_SUCCESS) {
        ret = agent.getXferStatus(treq);
        assert(ret >= 0);
    }
    std::cout << current_time() << __func__ << thread_id << "() Completed Sending Data using UCX backend. call nixlAgent.releaseXferReq()\n";
    agent.releaseXferReq(treq);
    std::cout << current_time() << __func__ << thread_id << "() call nixlAgent.invalidateLocalMD()\n";
    agent.invalidateLocalMD(&md_extra_params);
    show_nixl_opt_args("md_extra_params", &md_extra_params);

    std::cout << current_time() << __func__ << thread_id << "() Cleanup.. call nixlAgent.deregisterMem()\n";
    agent.deregisterMem(dram_for_ucx, extra_params);
    show_nixl_opt_args("extra_params", extra_params);

    std::cout << current_time() << __func__ << thread_id << "() Thead exit()\n";
}

static void runInitiator(const std::string &target_ip, int target_port, nixl_thread_sync_t sync_mode) {
    std::cout << current_time() << __func__ << "(" << target_ip << ", port:" << target_port << ")" << std::endl;

    nixlAgentConfig cfg(true, true, 0, sync_mode);
    cfg.pthrDelay = 1000*1000; //in us

    std::cout << "Starting Agent for initiator\n";
    nixlAgent agent(initiator, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    //nixl_b_params_t params = {
    //    { "num_workers", "4" },
    //};
    nixlBackendH *ucx;

    std::cout << current_time() << __func__ << "() calls nixlAgent.getPluginParams()" << std::endl;
    agent.getPluginParams("UCX", mems, params);
    std::cout << current_time() << __func__ << "() calls nixlAgent.createBackend()" << std::endl;
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    {
        nixl_b_params_t init1;
        nixl_mem_list_t mems1;
        nixl_status_t ret1;
        ret1 = agent.getBackendParams(ucx, mems1, init1);
        std::cout << current_time() << __func__ << "() called nixlAgent.getBackendParams()=" << ret1 << std::endl;
        printParams(init1, mems1);
    }

    SharedNotificationState shared_state;

    std::cout << current_time() << __func__ << "() create threads" << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
        threads.emplace_back(initiatorThread, std::ref(agent), &extra_params,
                             target_ip, target_port, i, std::ref(shared_state));

    std::cout << current_time() << __func__ << "() thread will join" << std::endl;
    for (auto &thread : threads)
        thread.join();
}

int main(int argc, char *argv[]) {
    std::cout << current_time() << __func__ << "()" << std::endl;

    /** Argument Parsing */
    if (argc < 4) {
        std::cout <<"Enter the required arguments" << std::endl;
        std::cout <<"  <Role:target,initiator> " <<"<Target IP> <Target Port>" << std::endl;
        exit(-1);
    }

    std::string role = std::string(argv[1]);
    const char  *target_ip   = argv[2];
    int         target_port = std::stoi(argv[3]);

    std::transform(role.begin(), role.end(), role.begin(), ::tolower);

    if (!role.compare(initiator) && !role.compare(target)) {
            std::cerr << "Invalid role. Use 'initiator' or 'target'."
                      << "Currently "<< role <<std::endl;
            return 1;
    }

    auto sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    if (argc == 5) {
        std::string sync_mode_str{argv[4]};
        std::transform(sync_mode_str.begin(), sync_mode_str.end(), sync_mode_str.begin(), ::tolower);
        if (sync_mode_str == "rw") {
            sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
            std::cout << "Using RW sync mode" << std::endl;
        } else if (sync_mode_str == "strict") {
            sync_mode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
            std::cout << "Using Strict sync mode" << std::endl;
        } else {
            std::cerr << "Invalid sync mode. Use 'rw' or 'strict'." << std::endl;
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
