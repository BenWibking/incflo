
define void @incflo_dss_flux_vjp(ptr nofree noundef readonly %0, ptr nofree noundef %1, double noundef %2, double noundef %3) alwaysinline {
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %6 = load double, ptr %5, align 8
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load double, ptr %7, align 8
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load double, ptr %9, align 8
  %11 = fsub double %8, %10
  %12 = fmul double %11, 2.000000e+00
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %14 = load double, ptr %13, align 8
  %15 = fsub double %14, %8
  %16 = fmul double %15, 2.000000e+00
  %17 = fsub double %14, %10
  %18 = fmul double %17, 5.000000e-01
  %19 = tail call noundef double @llvm.fabs.f64(double %12)
  %20 = tail call noundef double @llvm.fabs.f64(double %18)
  %21 = tail call noundef double @llvm.fabs.f64(double %16)
  %22 = fcmp olt double %20, %19
  %23 = select i1 %22, double %20, double %19
  %24 = fcmp olt double %21, %23
  %25 = select i1 %24, double %21, double %23
  %26 = fmul double %12, %16
  %27 = fcmp ogt double %26, 0.000000e+00
  %28 = select i1 %27, double %25, double 0.000000e+00
  %29 = fcmp ogt double %18, 0.000000e+00
  %30 = fneg double %28
  %31 = select i1 %29, double %28, double %30
  %32 = fmul double %31, 5.000000e-01
  %33 = fsub double %8, %32
  %34 = load double, ptr %0, align 8
  %35 = fsub double %10, %34
  %36 = fmul double %35, 2.000000e+00
  %37 = fsub double %8, %34
  %38 = fmul double %37, 5.000000e-01
  %39 = tail call noundef double @llvm.fabs.f64(double %36)
  %40 = tail call noundef double @llvm.fabs.f64(double %38)
  %41 = fcmp olt double %40, %39
  %42 = select i1 %41, double %40, double %39
  %43 = fcmp olt double %19, %42
  %44 = select i1 %43, double %19, double %42
  %45 = fmul double %12, %36
  %46 = fcmp ogt double %45, 0.000000e+00
  %47 = select i1 %46, double %44, double 0.000000e+00
  %48 = fcmp ogt double %38, 0.000000e+00
  %49 = fneg double %47
  %50 = select i1 %48, double %47, double %49
  %51 = fmul double %50, 5.000000e-01
  %52 = fadd double %10, %51
  %53 = fcmp ule double %6, 1.000000e-08
  br i1 %53, label %54, label %59

54:                                               ; preds = %4
  %55 = fcmp olt double %6, -1.000000e-08
  br i1 %55, label %59, label %56

56:                                               ; preds = %54
  %57 = fadd double %33, %52
  %58 = fmul double %57, 5.000000e-01
  br label %59

59:                                               ; preds = %56, %54, %4
  %60 = phi double [ %58, %56 ], [ %52, %4 ], [ %33, %54 ]
  %61 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %62 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %63 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %64 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %65 = fsub double %60, %8
  %66 = fmul double %3, %65
  %67 = fsub double %60, %10
  %68 = fmul double %2, %67
  %69 = fsub double %66, %68
  %70 = fneg fast double %2
  %71 = fmul fast double %6, %70
  %72 = fmul fast double %6, %3
  %73 = fadd fast double %71, %72
  %74 = fcmp uge double %6, -1.000000e-08
  %75 = and i1 %74, %53
  %76 = select fast i1 %74, double 0.000000e+00, double %73
  %77 = fmul fast double %73, 5.000000e-01
  %78 = select i1 %75, double %77, double 0.000000e+00
  %79 = select i1 %75, double %77, double -0.000000e+00
  %80 = fadd reassoc arcp contract afn double %76, %79
  %81 = select i1 %53, double %78, double %73
  %82 = fmul fast double %81, 5.000000e-01
  %83 = fneg fast double %82
  %84 = select fast i1 %48, double %82, double %83
  %85 = select fast i1 %46, double %84, double 0.000000e+00
  %86 = select fast i1 %43, double %85, double 0.000000e+00
  %87 = select fast i1 %43, double 0.000000e+00, double %85
  %88 = select fast i1 %41, double %87, double 0.000000e+00
  %89 = select fast i1 %41, double 0.000000e+00, double %87
  %90 = fcmp fast olt double %38, 0.000000e+00
  %91 = fneg fast double %88
  %92 = select fast i1 %90, double %91, double %88
  %93 = fcmp fast olt double %36, 0.000000e+00
  %94 = fneg fast double %89
  %95 = select fast i1 %93, double %94, double %89
  %96 = fmul fast double %92, 5.000000e-01
  %97 = fmul fast double %95, 2.000000e+00
  %98 = load double, ptr %1, align 8
  %99 = fadd fast double %96, %97
  %100 = fsub fast double %98, %99
  store double %100, ptr %1, align 8
  %101 = fmul fast double %80, -5.000000e-01
  %102 = fneg fast double %101
  %103 = select fast i1 %29, double %101, double %102
  %104 = select fast i1 %27, double %103, double 0.000000e+00
  %105 = select fast i1 %24, double %104, double 0.000000e+00
  %106 = select fast i1 %24, double 0.000000e+00, double %104
  %107 = select fast i1 %22, double %106, double 0.000000e+00
  %108 = select fast i1 %22, double 0.000000e+00, double %106
  %109 = fadd fast double %86, %108
  %110 = fcmp fast olt double %16, 0.000000e+00
  %111 = fneg fast double %105
  %112 = select fast i1 %110, double %111, double %105
  %113 = fcmp fast olt double %18, 0.000000e+00
  %114 = fneg fast double %107
  %115 = select fast i1 %113, double %114, double %107
  %116 = fcmp fast olt double %12, 0.000000e+00
  %117 = fneg fast double %109
  %118 = select fast i1 %116, double %117, double %109
  %119 = fmul fast double %115, 5.000000e-01
  %120 = fmul fast double %112, 2.000000e+00
  %121 = fadd fast double %119, %120
  %122 = load double, ptr %61, align 8
  %123 = fadd fast double %121, %122
  store double %123, ptr %61, align 8
  %124 = fmul fast double %118, 2.000000e+00
  %125 = load double, ptr %62, align 8
  %126 = fadd fast double %71, %119
  %127 = fadd fast double %81, %97
  %128 = fadd fast double %126, %124
  %129 = fsub fast double %127, %128
  %130 = fadd fast double %129, %125
  store double %130, ptr %62, align 8
  %131 = load double, ptr %63, align 8
  %132 = fadd fast double %72, %120
  %133 = fsub fast double %80, %132
  %134 = fadd fast double %133, %96
  %135 = fadd fast double %134, %124
  %136 = fadd fast double %135, %131
  store double %136, ptr %63, align 8
  %137 = load double, ptr %64, align 8
  %138 = fadd fast double %137, %69
  store double %138, ptr %64, align 8
  ret void
}

define void @incflo_dss_normal_vjp(ptr nofree noundef readonly %0, ptr nofree noundef %1, double noundef %2) alwaysinline {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = load double, ptr %4, align 8
  %6 = load double, ptr %0, align 8
  %7 = fsub double %5, %6
  %8 = fmul double %7, 2.000000e+00
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %10 = load double, ptr %9, align 8
  %11 = fsub double %10, %5
  %12 = fmul double %11, 2.000000e+00
  %13 = fsub double %10, %6
  %14 = fmul double %13, 5.000000e-01
  %15 = tail call noundef double @llvm.fabs.f64(double %8)
  %16 = tail call noundef double @llvm.fabs.f64(double %14)
  %17 = tail call noundef double @llvm.fabs.f64(double %12)
  %18 = fcmp olt double %16, %15
  %19 = select i1 %18, double %16, double %15
  %20 = fcmp olt double %17, %19
  %21 = select i1 %20, double %17, double %19
  %22 = fmul double %8, %12
  %23 = fcmp ogt double %22, 0.000000e+00
  %24 = select i1 %23, double %21, double 0.000000e+00
  %25 = fcmp ogt double %14, 0.000000e+00
  %26 = fneg double %24
  %27 = select i1 %25, double %24, double %26
  %28 = fmul double %27, 5.000000e-01
  %29 = fadd double %5, %28
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %31 = load double, ptr %30, align 8
  %32 = fsub double %31, %10
  %33 = fmul double %32, 2.000000e+00
  %34 = fsub double %31, %5
  %35 = fmul double %34, 5.000000e-01
  %36 = tail call noundef double @llvm.fabs.f64(double %35)
  %37 = tail call noundef double @llvm.fabs.f64(double %33)
  %38 = fcmp olt double %36, %17
  %39 = select i1 %38, double %36, double %17
  %40 = fcmp olt double %37, %39
  %41 = select i1 %40, double %37, double %39
  %42 = fmul double %12, %33
  %43 = fcmp ogt double %42, 0.000000e+00
  %44 = select i1 %43, double %41, double 0.000000e+00
  %45 = fcmp ogt double %35, 0.000000e+00
  %46 = fneg double %44
  %47 = select i1 %45, double %44, double %46
  %48 = fmul double %47, 5.000000e-01
  %49 = fsub double %10, %48
  %50 = fadd double %29, %49
  %51 = fmul double %50, 5.000000e-01
  %52 = fcmp oge double %29, 0.000000e+00
  %53 = fcmp ole double %49, 0.000000e+00
  %54 = select i1 %52, i1 true, i1 %53
  br i1 %54, label %55, label %59

55:                                               ; preds = %3
  %56 = fcmp ult double %51, 1.000000e-08
  br i1 %56, label %57, label %60

57:                                               ; preds = %55
  %58 = fcmp ugt double %51, -1.000000e-08
  br i1 %58, label %59, label %60

59:                                               ; preds = %57, %3
  br label %60

60:                                               ; preds = %59, %57, %55
  %61 = phi double [ 0.000000e+00, %59 ], [ %2, %55 ], [ 0.000000e+00, %57 ]
  %62 = phi double [ 0.000000e+00, %59 ], [ 0.000000e+00, %55 ], [ %2, %57 ]
  %63 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %64 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %65 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %66 = fmul fast double %62, -5.000000e-01
  %67 = fneg fast double %66
  %68 = select fast i1 %45, double %66, double %67
  %69 = select fast i1 %43, double %68, double 0.000000e+00
  %70 = select fast i1 %40, double %69, double 0.000000e+00
  %71 = select fast i1 %40, double 0.000000e+00, double %69
  %72 = select fast i1 %38, double %71, double 0.000000e+00
  %73 = select fast i1 %38, double 0.000000e+00, double %71
  %74 = fcmp fast olt double %33, 0.000000e+00
  %75 = fneg fast double %70
  %76 = select fast i1 %74, double %75, double %70
  %77 = fcmp fast olt double %35, 0.000000e+00
  %78 = fneg fast double %72
  %79 = select fast i1 %77, double %78, double %72
  %80 = fmul fast double %79, 5.000000e-01
  %81 = fmul fast double %76, 2.000000e+00
  %82 = load double, ptr %63, align 8
  %83 = fadd fast double %81, %82
  %84 = fadd fast double %83, %80
  store double %84, ptr %63, align 8
  %85 = fmul fast double %61, 5.000000e-01
  %86 = fneg fast double %85
  %87 = select fast i1 %25, double %85, double %86
  %88 = select fast i1 %23, double %87, double 0.000000e+00
  %89 = select fast i1 %20, double 0.000000e+00, double %88
  %90 = select fast i1 %20, double %88, double 0.000000e+00
  %91 = fadd fast double %73, %90
  %92 = select fast i1 %18, double %89, double 0.000000e+00
  %93 = select fast i1 %18, double 0.000000e+00, double %89
  %94 = fcmp fast olt double %12, 0.000000e+00
  %95 = fneg fast double %91
  %96 = select fast i1 %94, double %95, double %91
  %97 = fcmp fast olt double %14, 0.000000e+00
  %98 = fneg fast double %92
  %99 = select fast i1 %97, double %98, double %92
  %100 = fcmp fast olt double %8, 0.000000e+00
  %101 = fneg fast double %93
  %102 = select fast i1 %100, double %101, double %93
  %103 = fmul fast double %99, 5.000000e-01
  %104 = fmul fast double %96, 2.000000e+00
  %105 = load double, ptr %64, align 8
  %106 = fadd fast double %105, %62
  %107 = fsub fast double %106, %81
  %108 = fadd fast double %107, %103
  %109 = fadd fast double %108, %104
  store double %109, ptr %64, align 8
  %110 = fmul fast double %102, 2.000000e+00
  %111 = load double, ptr %1, align 8
  %112 = fadd fast double %103, %110
  %113 = fsub fast double %111, %112
  store double %113, ptr %1, align 8
  %114 = load double, ptr %65, align 8
  %115 = fadd fast double %110, %61
  %116 = fadd fast double %115, %114
  %117 = fadd fast double %80, %104
  %118 = fsub fast double %116, %117
  store double %118, ptr %65, align 8
  ret void
}

declare double @llvm.fabs.f64(double)
