"""Render an automatic completion as a distinct, source-bound review revision."""
import argparse,json
from pathlib import Path


def render(template, result, image_prefix):
    page=template
    begin=page.index('const data=');end=page.index(';const $',begin)
    page=page[:begin]+'const data='+json.dumps(result,ensure_ascii=False).replace('<','\\u003c')+page[end:]
    begin=page.index('// Shared by independent and assisted boards')
    end=page.index('</script>',begin)
    page=page[:begin]+Path(__file__).with_name('joint_annotation_workflow.js').read_text()+page[end:]
    page=page.replace("setAttribute('href',f.file)","setAttribute('href',"+json.dumps(image_prefix)+"+f.file)")
    page=page.replace("data.frames.forEach", "$('annotator').value=data.annotator_id||'';data.frames.forEach",1)
    page=page.replace("point.reviewed?'#22ee88'", "point.reviewed&&point.review_actor!=='automatic'?'#22ee88'")
    page=page.replace("p.reviewed?'已复核':'待复核'", "p.review_actor==='automatic'?'自动处理（未人工确认）':p.reviewed?'人工已复核':'待复核'")
    page=page.replace("已复核：'+Object.values(data.labels).filter(p=>p.reviewed).length", "人工已复核：'+Object.values(data.labels).filter(p=>p.reviewed&&p.review_actor!=='automatic').length+' / 自动处理：'+Object.values(data.labels).filter(p=>p.review_actor==='automatic').length")
    page=page.replace('下一待审核项','下一留空项')
    page=page.replace("if(data.labels[item.key]?.reviewed===true)continue", "if(data.labels[item.key]?.visible!==false)continue")
    # Start after the current item and wrap, so repeated clicks visit all blanks.
    page=page.replace('for(const item of data.review_queue){if(data.labels[item.key]?.visible', "for(const item of [...data.review_queue.slice(data.review_queue.findIndex(x=>x.key===key())+1),...data.review_queue.slice(0,data.review_queue.findIndex(x=>x.key===key())+1)]){if(data.labels[item.key]?.visible")
    page=page.replace("!p.reviewed&&p.origin==='model'", "p.visible&&(!p.reviewed||p.review_actor==='automatic')")
    page=page.replace("p.origin='human_bulk_accepted_model';", "p.review_actor='human';p.origin='human_bulk_accepted_model';")
    page=page.replace("p.reviewed=true;p.origin='human_accepted_model'", "p.reviewed=true;p.review_actor='human';p.origin='human_accepted_model'")
    page=page.replace('全部关节点已审核（含不可辨认项）','全部人工已确认（自动处理不计入）')
    page=page.replace('<h1>50.03 · 模型预标注复核</h1>', '<h1>50.03 · 自动优化复核结果</h1><p>人工原样保留 144 项；剩余 1856 项自动处理：1737 项候选、119 项留空。连同人工不可辨认 14 项，共 133 项留空。绿色为人工、橙色为自动、青色为选中点。自动结果不作为独立准确率真值。</p><p><a href="optimized_review.json">下载完整结果</a> · <a href="optimization_report.json">逐项处理理由</a> · <a href="received_latest.json">收到的人工原稿</a></p>')
    page=page.replace('href="prelabel_audit.json"', 'href="'+image_prefix+'prelabel_audit.json"')
    page=page.replace('需要全部审核：每项都须确认或标记不可辨认。', '自动处理已完成；留空点表示证据不足，不强制补标。')
    page=page.replace('缺失点需人工处理。','自动结果可按需人工修改。')
    page=page.replace('绿色为已复核','绿色为人工复核')
    return page


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('template','review','image-prefix','output'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args()
    Path(a.output).write_text(render(Path(a.template).read_text(),json.loads(Path(a.review).read_text()),a.image_prefix))


if __name__=='__main__':main()
