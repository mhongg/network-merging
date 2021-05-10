from .smart_coord import smart_coord_upan, smart_coord_fpan


def smart_coordinator_upan(args, model1, model2, upan, device, test_loader):
    result = []
    print(f"PAN type: {args.upan_type}")
    test_loss, acc = smart_coord_upan(
        args, model1, model2, upan, device, test_loader
    )
    result.append({"test_loss": test_loss, "acc": acc})
    return result

def smart_coordinator_fpan(args, model1, model2, fpan, device, test_loader):
    result = []
    test_loss, acc = smart_coord_fpan(
        args, model1, model2, fpan, device, test_loader
    )
    result.append({"test_loss": test_loss, "acc": acc})
    return result