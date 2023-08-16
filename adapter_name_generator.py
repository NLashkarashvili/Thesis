def get_adapter_name(args, sequence_model):
    

    adapter_name = ''
    rank = ''
    bias_tuning = False
    if args.lora_adapter:
        rank = args.lora_rank
    if args.bitfit:
        print(args.bitfit)
        bias_tuning = True

    if args.adapter_name == 'none' and args.lora_adapter and not args.bitfit:
        if args.use_weighted_layer_sum:
            adapter_name ='lora'
            sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
        else:
            adapter_name = 'only_lora'
            sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
    elif args.adapter_name != 'none':
        adapter_name += args.adapter_name 
        if args.adapter_layer_list:
            adapter_name += "".join([str(i) for i in args.adapter_layer_list])
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)


    if args.bitfit:
        if args.use_weighted_layer_sum and args.lora_adapter:
            adapter_name = 'bitfit+lora+ws'
            sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
        elif args.use_weighted_layer_sum:
            adapter_name = 'bitfit+ws'
            sequence_model.freeze_base_model(bias_tuning=bias_tuning)
        elif args.lora_adapter:
            adapter_name = 'bitfit+lora'
            sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
        else:
            adapter_name = 'bitfit'
            sequence_model.freeze_base_model(bias_tuning=bias_tuning)
    elif args.adapter_name!='none' or args.add_adapter:
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
        
    if args.adapter_name == 'none' and not (args.lora_adapter \
            or args.prefix_tuning or args.add_adapter or args.bitfit):
        if args.use_weighted_layer_sum:
            adapter_name='superb'
            sequence_model.freeze_base_model(bias_tuning=bias_tuning)
        elif args.tune_last_block:
            adapter_name='last_block'
            sequence_model.freeze_base_model(tune_last_block = True)
        elif args.fine_tune:
            adapter_name = "fine_tune"
            sequence_model.freeze_feature_encoder()
            pass


    if args.output_adapter and args.mh_adapter:
        adapter_name = "mh_output" + adapter_name + str(args.adapter_dim)
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
    elif args.output_adapter:
        adapter_name = "output" + adapter_name + str(args.adapter_dim)
        if args.use_weighted_layer_sum and "ws" not in adapter_name:
            adapter_name += "_ws"
        if args.lora_adapter and 'lora' not in adapter_name:
            adapter_name += "_lora"
        if args.bitfit and 'bitfit' not in adapter_name:
            adapter_name += "_bitfit"
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
    elif args.mh_adapter:
        adapter_name = "mh" + adapter_name + str(args.adapter_dim)
        if args.use_weighted_layer_sum:
            adapter_name += "_ws"
        if args.lora_adapter:
            adapter_name += "_lora"
        if args.bitfit:
            adapter_name += "_bitfit"
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
    if args.add_adapter:
        if len(adapter_name) == 0:
            st_str = ''
        else:
            st_str = '_'
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
        adapter_name += st_str + "encoder"
        
    if args.encoder_tuning: ## TODO Change this 
        if len(adapter_name) == 0:
            st_str = ''
        else:
            st_str = '_'
        adapter_name += st_str + "encoder_tuning"
        sequence_model.freeze_transformer_blocks()
        if args.use_weighted_layer_sum and "ws" not in adapter_name:
            adapter_name += "_ws"
        if args.last_3_enc:
            adapter_name += "_last3"
        
    if args.cnn_multiplier:
        adapter_name = "cnn_mul"
    
    if args.trans_multiplier:
        if len(adapter_name) == 0:
            sequence_model.freeze_base_model(bias_tuning=bias_tuning)
            st_str = ''
        else:
            st_str = '_'
        adapter_name += st_str + "trans_mult"
        if args.every_hidden_state:
            adapter_name += "_all"

        # if args.cnn_multiplier:
            # adapter_name += "cnn"
    if args.use_vert_layer_sum:
        if len(adapter_name) == 0:
            sequence_model.freeze_base_model(bias_tuning=bias_tuning)
            st_str = ''
        else:
            st_str = '_'
        adapter_name += st_str + "vs"
        sequence_model.freeze_feature_encoder()
        
    if args.conv_lora:
        if len(adapter_name) == 0 :
            st_str = ''
        else:
            st_str = '_'
        adapter_name +="conv_lora" + str(args.conv_r)
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
        
    if args.bottleneck_cnn:
        rank_str = str(args.bconv_d)
        encoder_cnt = str(args.n_enc_adapters)
        if len(adapter_name) == 0 :
            st_str = ''
        else:
            st_str = '_'

        adapter_name += st_str + "bottleneck_cnn" + rank_str + "_" + encoder_cnt
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
    
    if args.depthwise_cnn:
        if len(adapter_name) == 0 :
            st_str = ''
        else:
            st_str = '_'
        adapter_name += st_str + "depthwise_cnn"
        sequence_model.freeze_exclude_prompt(bias_tuning=bias_tuning)
        
    if len(adapter_name) == 0:
        adapter_name='transformers'
        sequence_model.freeze_base_model(bias_tuning=bias_tuning)



    return adapter_name, sequence_model