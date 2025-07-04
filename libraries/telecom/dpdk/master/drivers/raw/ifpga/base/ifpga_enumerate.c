/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2018 Intel Corporation
 */

#include "opae_hw_api.h"
#include "ifpga_api.h"

#include "ifpga_hw.h"
#include "ifpga_enumerate.h"
#include "ifpga_feature_dev.h"

struct build_feature_devs_info {
	struct opae_adapter_data_pci *pci_data;

	struct ifpga_afu_info *acc_info;

	void *fiu;
	enum fpga_id_type lwrrent_type;
	int lwrrent_port_id;

	void *ioaddr;
	void *ioend;
	uint64_t phys_addr;
	int lwrrent_bar;

	void *pfme_hdr;

	struct ifpga_hw *hw;
};

static int feature_revision(void __iomem *start)
{
	struct feature_header header;

	header.csr = readq(start);

	return header.revision;
}

static u32 feature_size(void __iomem *start)
{
	struct feature_header header;

	header.csr = readq(start);

	/*the size of private feature is 4KB aligned*/
	return header.next_header_offset ? header.next_header_offset:4096;
}

static u64 feature_id(void __iomem *start)
{
	struct feature_header header;

	header.csr = readq(start);

	switch (header.type) {
	case FEATURE_TYPE_FIU:
		return FEATURE_ID_FIU_HEADER;
	case FEATURE_TYPE_PRIVATE:
		return header.id;
	case FEATURE_TYPE_AFU:
		return FEATURE_ID_AFU;
	}

	WARN_ON(1);
	return 0;
}

static int
build_info_add_sub_feature(struct build_feature_devs_info *binfo,
		void __iomem *start, u64 fid, unsigned int size,
		unsigned int vec_start,
		unsigned int vec_cnt)
{
	struct ifpga_hw *hw = binfo->hw;
	struct ifpga_feature *feature = NULL;
	struct feature_irq_ctx *ctx = NULL;
	int port_id, ret = 0;
	unsigned int i;

	fid = fid?fid:feature_id(start);
	size = size?size:feature_size(start);

	feature = opae_malloc(sizeof(struct ifpga_feature));
	if (!feature)
		return -ENOMEM;

	feature->state = IFPGA_FEATURE_ATTACHED;
	feature->addr = start;
	feature->id = fid;
	feature->size = size;
	feature->revision = feature_revision(start);
	feature->phys_addr = binfo->phys_addr +
				((u8 *)start - (u8 *)binfo->ioaddr);
	feature->vec_start = vec_start;
	feature->vec_cnt = vec_cnt;

	dev_debug(binfo, "%s: id=0x%llx, phys_addr=0x%llx, size=%u\n",
			__func__, (unsigned long long)feature->id,
			(unsigned long long)feature->phys_addr, size);

	if (vec_cnt) {
		if (vec_start + vec_cnt <= vec_start)
			return -EILWAL;

		ctx = zmalloc(sizeof(*ctx) * vec_cnt);
		if (!ctx)
			return -ENOMEM;

		for (i = 0; i < vec_cnt; i++) {
			ctx[i].eventfd = -1;
			ctx[i].idx = vec_start + i;
		}
	}

	feature->ctx = ctx;
	feature->ctx_num = vec_cnt;
	feature->vfio_dev_fd = binfo->pci_data->vfio_dev_fd;

	if (binfo->lwrrent_type == FME_ID) {
		feature->parent = &hw->fme;
		feature->type = FEATURE_FME_TYPE;
		feature->name = get_fme_feature_name(fid);
		TAILQ_INSERT_TAIL(&hw->fme.feature_list, feature, next);
	} else if (binfo->lwrrent_type == PORT_ID) {
		port_id = binfo->lwrrent_port_id;
		feature->parent = &hw->port[port_id];
		feature->type = FEATURE_PORT_TYPE;
		feature->name = get_port_feature_name(fid);
		TAILQ_INSERT_TAIL(&hw->port[port_id].feature_list,
				feature, next);
	} else {
		return -EFAULT;
	}
	return ret;
}

static int
create_feature_instance(struct build_feature_devs_info *binfo,
			void __iomem *start, u64 fid,
			unsigned int size, unsigned int vec_start,
			unsigned int vec_cnt)
{
	return build_info_add_sub_feature(binfo, start, fid, size, vec_start,
			vec_cnt);
}

/*
 * UAFU GUID is dynamic as it can be changed after FME downloads different
 * Green Bitstream to the port, so we treat the unknown GUIDs which are
 * attached on port's feature list as UAFU.
 */
static bool feature_is_UAFU(struct build_feature_devs_info *binfo)
{
	if (binfo->lwrrent_type != PORT_ID)
		return false;

	return true;
}

static int parse_feature_port_uafu(struct build_feature_devs_info *binfo,
				   struct feature_header *hdr)
{
	u64 id = PORT_FEATURE_ID_UAFU;
	struct ifpga_afu_info *info;
	void *start = (void *)hdr;
	struct feature_port_header *port_hdr = binfo->ioaddr;
	struct feature_port_capability capability;
	int ret;
	int size;

	capability.csr = readq(&port_hdr->capability);

	size = capability.mmio_size << 10;

	ret = create_feature_instance(binfo, hdr, id, size, 0, 0);
	if (ret)
		return ret;

	info = opae_malloc(sizeof(*info));
	if (!info)
		return -ENOMEM;

	info->region[0].addr = start;
	info->region[0].phys_addr = binfo->phys_addr +
			(uint8_t *)start - (uint8_t *)binfo->ioaddr;
	info->region[0].len = size;
	info->num_regions = 1;

	binfo->acc_info = info;

	return ret;
}

static int parse_feature_afus(struct build_feature_devs_info *binfo,
			      struct feature_header *hdr)
{
	int ret;
	struct feature_afu_header *afu_hdr, header;
	u8 __iomem *start;
	u8 __iomem *end = binfo->ioend;

	start = (u8 __iomem *)hdr;
	for (; start < end; start += header.next_afu) {
		if ((unsigned int)(end - start) <
			(unsigned int)(sizeof(*afu_hdr) + sizeof(*hdr)))
			return -EILWAL;

		hdr = (struct feature_header *)start;
		afu_hdr = (struct feature_afu_header *)(hdr + 1);
		header.csr = readq(&afu_hdr->csr);

		if (feature_is_UAFU(binfo)) {
			ret = parse_feature_port_uafu(binfo, hdr);
			if (ret)
				return ret;
		}

		if (!header.next_afu)
			break;
	}

	return 0;
}

/* create and register proper private data */
static int build_info_commit_dev(struct build_feature_devs_info *binfo)
{
	struct ifpga_afu_info *info = binfo->acc_info;
	struct ifpga_hw *hw = binfo->hw;
	struct opae_manager *mgr;
	struct opae_bridge *br;
	struct opae_accelerator *acc;
	struct ifpga_port_hw *port;
	struct ifpga_fme_hw *fme;
	struct ifpga_feature *feature;

	if (!binfo->fiu)
		return 0;

	if (binfo->lwrrent_type == PORT_ID) {
		/* return error if no valid acc info data structure */
		if (!info)
			return -EFAULT;

		br = opae_bridge_alloc(hw->adapter->name, &ifpga_br_ops,
				       binfo->fiu);
		if (!br)
			return -ENOMEM;

		br->id = binfo->lwrrent_port_id;

		/* update irq info */
		port = &hw->port[binfo->lwrrent_port_id];
		feature = get_feature_by_id(&port->feature_list,
				PORT_FEATURE_ID_UINT);
		if (feature)
			info->num_irqs = feature->vec_cnt;

		acc = opae_accelerator_alloc(hw->adapter->name,
					     &ifpga_acc_ops, info);
		if (!acc) {
			opae_bridge_free(br);
			return -ENOMEM;
		}

		acc->br = br;
		if (hw->adapter->mgr)
			acc->mgr = hw->adapter->mgr;
		acc->index = br->id;

		fme = &hw->fme;
		fme->nums_acc_region = info->num_regions;

		opae_adapter_add_acc(hw->adapter, acc);

	} else if (binfo->lwrrent_type == FME_ID) {
		mgr = opae_manager_alloc(hw->adapter->name, &ifpga_mgr_ops,
				&ifpga_mgr_network_ops, binfo->fiu);
		if (!mgr)
			return -ENOMEM;

		mgr->adapter = hw->adapter;
		hw->adapter->mgr = mgr;
	}

	binfo->fiu = NULL;

	return 0;
}

static int
build_info_create_dev(struct build_feature_devs_info *binfo,
		      enum fpga_id_type type, unsigned int index)
{
	int ret;

	ret = build_info_commit_dev(binfo);
	if (ret)
		return ret;

	binfo->lwrrent_type = type;

	if (type == FME_ID) {
		binfo->fiu = &binfo->hw->fme;
	} else if (type == PORT_ID) {
		binfo->fiu = &binfo->hw->port[index];
		binfo->lwrrent_port_id = index;
	}

	return 0;
}

static int parse_feature_fme(struct build_feature_devs_info *binfo,
			     struct feature_header *start)
{
	struct ifpga_hw *hw = binfo->hw;
	struct ifpga_fme_hw *fme = &hw->fme;
	int ret;

	ret = build_info_create_dev(binfo, FME_ID, 0);
	if (ret)
		return ret;

	/* Update FME states */
	fme->state = IFPGA_FME_IMPLEMENTED;
	fme->parent = hw;
	TAILQ_INIT(&fme->feature_list);
	spinlock_init(&fme->lock);

	return create_feature_instance(binfo, start, 0, 0, 0, 0);
}

static int parse_feature_port(struct build_feature_devs_info *binfo,
			      void __iomem *start)
{
	struct feature_port_header *port_hdr;
	struct feature_port_capability capability;
	struct ifpga_hw *hw = binfo->hw;
	struct ifpga_port_hw *port;
	unsigned int port_id;
	int ret;

	/* Get current port's id */
	port_hdr = (struct feature_port_header *)start;
	capability.csr = readq(&port_hdr->capability);
	port_id = capability.port_number;

	ret = build_info_create_dev(binfo, PORT_ID, port_id);
	if (ret)
		return ret;

	/*found a Port device*/
	port = &hw->port[port_id];
	port->port_id = binfo->lwrrent_port_id;
	port->parent = hw;
	port->state = IFPGA_PORT_ATTACHED;
	spinlock_init(&port->lock);
	TAILQ_INIT(&port->feature_list);

	return create_feature_instance(binfo, start, 0, 0, 0, 0);
}

static void enable_port_uafu(struct build_feature_devs_info *binfo,
			     void __iomem *start)
{
	struct ifpga_port_hw *port = &binfo->hw->port[binfo->lwrrent_port_id];

	UNUSED(start);

	fpga_port_reset(port);
}

static int parse_feature_fiu(struct build_feature_devs_info *binfo,
			     struct feature_header *hdr)
{
	struct feature_header header;
	struct feature_fiu_header *fiu_hdr, fiu_header;
	u8 __iomem *start = (u8 __iomem *)hdr;
	int ret;

	header.csr = readq(hdr);

	switch (header.id) {
	case FEATURE_FIU_ID_FME:
		ret = parse_feature_fme(binfo, hdr);
		binfo->pfme_hdr = hdr;
		if (ret)
			return ret;
		break;
	case FEATURE_FIU_ID_PORT:
		ret = parse_feature_port(binfo, hdr);
		enable_port_uafu(binfo, hdr);
		if (ret)
			return ret;

		/* Check Port FIU's next_afu pointer to User AFU DFH */
		fiu_hdr = (struct feature_fiu_header *)(hdr + 1);
		fiu_header.csr = readq(&fiu_hdr->csr);

		if (fiu_header.next_afu) {
			start += fiu_header.next_afu;
			ret = parse_feature_afus(binfo,
						(struct feature_header *)start);
			if (ret)
				return ret;
		} else {
			dev_info(binfo, "No AFUs detected on Port\n");
		}

		break;
	default:
		dev_info(binfo, "FIU TYPE %d is not supported yet.\n",
			 header.id);
	}

	return 0;
}

static void parse_feature_irqs(struct build_feature_devs_info *binfo,
		void __iomem *start, unsigned int *vec_start,
		unsigned int *vec_cnt)
{
	UNUSED(binfo);
	u64 id;

	id = feature_id(start);

	if (id == PORT_FEATURE_ID_UINT) {
		struct feature_port_uint *port_uint = start;
		struct feature_port_uint_cap uint_cap;

		uint_cap.csr = readq(&port_uint->capability);
		if (uint_cap.intr_num) {
			*vec_start = uint_cap.first_vec_num;
			*vec_cnt = uint_cap.intr_num;
		} else {
			dev_debug(binfo, "UAFU doesn't support interrupt\n");
		}
	} else if (id == PORT_FEATURE_ID_ERROR) {
		struct feature_port_error *port_err = start;
		struct feature_port_err_capability port_err_cap;

		port_err_cap.csr = readq(&port_err->error_capability);
		if (port_err_cap.support_intr) {
			*vec_start = port_err_cap.intr_vector_num;
			*vec_cnt = 1;
		} else {
			dev_debug(&binfo, "Port error doesn't support interrupt\n");
		}

	} else if (id == FME_FEATURE_ID_GLOBAL_ERR) {
		struct feature_fme_err *fme_err = start;
		struct feature_fme_error_capability fme_err_cap;

		fme_err_cap.csr = readq(&fme_err->fme_err_capability);
		if (fme_err_cap.support_intr) {
			*vec_start = fme_err_cap.intr_vector_num;
			*vec_cnt = 1;
		} else {
			dev_debug(&binfo, "FME error doesn't support interrupt\n");
		}
	}
}

static int parse_feature_fme_private(struct build_feature_devs_info *binfo,
				     struct feature_header *hdr)
{
	unsigned int vec_start = 0;
	unsigned int vec_cnt = 0;

	parse_feature_irqs(binfo, hdr, &vec_start, &vec_cnt);

	return create_feature_instance(binfo, hdr, 0, 0, vec_start, vec_cnt);
}

static int parse_feature_port_private(struct build_feature_devs_info *binfo,
				      struct feature_header *hdr)
{
	unsigned int vec_start = 0;
	unsigned int vec_cnt = 0;

	parse_feature_irqs(binfo, hdr, &vec_start, &vec_cnt);

	return create_feature_instance(binfo, hdr, 0, 0, vec_start, vec_cnt);
}

static int parse_feature_private(struct build_feature_devs_info *binfo,
				 struct feature_header *hdr)
{
	struct feature_header header;

	header.csr = readq(hdr);

	switch (binfo->lwrrent_type) {
	case FME_ID:
		return parse_feature_fme_private(binfo, hdr);
	case PORT_ID:
		return parse_feature_port_private(binfo, hdr);
	default:
		dev_err(binfo, "private feature %x belonging to AFU %d (unknown_type) is not supported yet.\n",
			header.id, binfo->lwrrent_type);
	}
	return 0;
}

static int parse_feature(struct build_feature_devs_info *binfo,
			 struct feature_header *hdr)
{
	struct feature_header header;
	int ret = 0;

	header.csr = readq(hdr);

	switch (header.type) {
	case FEATURE_TYPE_AFU:
		ret = parse_feature_afus(binfo, hdr);
		break;
	case FEATURE_TYPE_PRIVATE:
		ret = parse_feature_private(binfo, hdr);
		break;
	case FEATURE_TYPE_FIU:
		ret = parse_feature_fiu(binfo, hdr);
		break;
	default:
		dev_err(binfo, "Feature Type %x is not supported.\n",
			hdr->type);
	};

	return ret;
}

static int
parse_feature_list(struct build_feature_devs_info *binfo, u8 __iomem *start)
{
	struct feature_header *hdr, header;
	u8 __iomem *end = (u8 __iomem *)binfo->ioend;
	int ret = 0;

	for (; start < end; start += header.next_header_offset) {
		if ((unsigned int)(end - start) < (unsigned int)sizeof(*hdr)) {
			dev_err(binfo, "The region is too small to contain a feature.\n");
			ret =  -EILWAL;
			break;
		}

		hdr = (struct feature_header *)start;
		header.csr = readq(hdr);

		dev_debug(binfo, "%s: address=0x%p, val=0x%llx, header.id=0x%x, header.next_offset=0x%x, header.eol=0x%x, header.type=0x%x\n",
			__func__, hdr, (unsigned long long)header.csr,
			header.id, header.next_header_offset,
			header.end_of_list, header.type);

		ret = parse_feature(binfo, hdr);
		if (ret)
			return ret;

		if (header.end_of_list || !header.next_header_offset)
			break;
	}

	return build_info_commit_dev(binfo);
}

/* switch the memory mapping to BAR# @bar */
static int parse_switch_to(struct build_feature_devs_info *binfo, int bar)
{
	struct opae_adapter_data_pci *pci_data = binfo->pci_data;

	if (!pci_data->region[bar].addr)
		return -ENOMEM;

	binfo->ioaddr = pci_data->region[bar].addr;
	binfo->ioend = (u8 __iomem *)binfo->ioaddr + pci_data->region[bar].len;
	binfo->phys_addr = pci_data->region[bar].phys_addr;
	binfo->lwrrent_bar = bar;

	return 0;
}

static int parse_ports_from_fme(struct build_feature_devs_info *binfo)
{
	struct feature_fme_header *fme_hdr;
	struct feature_fme_port port;
	int i = 0, ret = 0;

	if (!binfo->pfme_hdr) {
		dev_info(binfo,  "VF is detected.\n");
		return ret;
	}

	fme_hdr = binfo->pfme_hdr;

	do {
		port.csr = readq(&fme_hdr->port[i]);
		if (!port.port_implemented)
			break;

		/* skip port which only could be accessed via VF */
		if (port.afu_access_control == FME_AFU_ACCESS_VF)
			continue;

		ret = parse_switch_to(binfo, port.port_bar);
		if (ret)
			break;

		ret = parse_feature_list(binfo,
					 (u8 __iomem *)binfo->ioaddr +
					  port.port_offset);
		if (ret)
			break;
	} while (++i < MAX_FPGA_PORT_NUM);

	return ret;
}

static struct build_feature_devs_info *
build_info_alloc_and_init(struct ifpga_hw *hw)
{
	struct build_feature_devs_info *binfo;

	binfo = zmalloc(sizeof(*binfo));
	if (!binfo)
		return binfo;

	binfo->hw = hw;
	binfo->pci_data = hw->pci_data;

	/* fpga feature list starts from BAR 0 */
	if (parse_switch_to(binfo, 0)) {
		free(binfo);
		return NULL;
	}

	return binfo;
}

static void build_info_free(struct build_feature_devs_info *binfo)
{
	free(binfo);
}

static void ifpga_print_device_feature_list(struct ifpga_hw *hw)
{
	struct ifpga_fme_hw *fme = &hw->fme;
	struct ifpga_port_hw *port;
	struct ifpga_feature *feature;
	int i;

	dev_info(hw, "found fme_device, is in PF: %s\n",
		 is_ifpga_hw_pf(hw) ? "yes" : "no");

	ifpga_for_each_fme_feature(fme, feature) {
		if (feature->state != IFPGA_FEATURE_ATTACHED)
			continue;

		dev_info(hw, "%12s:	%p - %p  - paddr: 0x%lx\n",
			 feature->name, feature->addr,
			 feature->addr + feature->size - 1,
			 (unsigned long)feature->phys_addr);

	}

	for (i = 0; i < MAX_FPGA_PORT_NUM; i++) {
		port = &hw->port[i];

		if (port->state != IFPGA_PORT_ATTACHED)
			continue;

		dev_info(hw, "port device: %d\n", port->port_id);

		ifpga_for_each_port_feature(port, feature) {
			if (feature->state != IFPGA_FEATURE_ATTACHED)
				continue;

			dev_info(hw, "%12s:	%p - %p  - paddr:0x%lx\n",
				 feature->name,
				 feature->addr,
				 feature->addr +
				 feature->size - 1,
				 (unsigned long)feature->phys_addr);
		}

	}
}

int ifpga_bus_enumerate(struct ifpga_hw *hw)
{
	struct build_feature_devs_info *binfo;
	int ret;

	binfo = build_info_alloc_and_init(hw);
	if (!binfo)
		return -ENOMEM;

	ret = parse_feature_list(binfo, binfo->ioaddr);
	if (ret)
		goto exit;

	ret = parse_ports_from_fme(binfo);
	if (ret)
		goto exit;

	ifpga_print_device_feature_list(hw);

exit:
	build_info_free(binfo);
	return ret;
}

int ifpga_bus_init(struct ifpga_hw *hw)
{
	int i;
	struct ifpga_port_hw *port;

	fme_hw_init(&hw->fme);
	for (i = 0; i < MAX_FPGA_PORT_NUM; i++) {
		port = &hw->port[i];
		port_hw_init(port);
	}

	return 0;
}

int ifpga_bus_uinit(struct ifpga_hw *hw)
{
	int i;
	struct ifpga_port_hw *port;

	if (hw) {
		fme_hw_uinit(&hw->fme);
		for (i = 0; i < MAX_FPGA_PORT_NUM; i++) {
			port = &hw->port[i];
			port_hw_uinit(port);
		}
	}

	return 0;
}
