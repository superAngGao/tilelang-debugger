"""Open actual file:// reports in an installed browser, with network disabled."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from tilelang_debugger.reporting.render import html


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reports',required=True)
    parser.add_argument('--browser',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root,output=Path(args.reports).resolve(),Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    result=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(executable_path=args.browser,headless=True)
        context=browser.new_context(viewport=dict(width=1440,height=1000),offline=True)
        for name in ('mixed','fragment','strided','no-reference'):
            page=context.new_page();errors=[]
            page.on('pageerror',lambda e: errors.append(str(e)))
            page.goto((root/name/'report.html').as_uri())
            page.wait_for_selector('.point')
            assert page.locator('h1').inner_text()=='源码观察与调试报告'
            count=page.locator('.point').count()
            for i in range(count):
                page.locator('.point').nth(i).click()
                for tab in ('values','accesses','compiler','context'):
                    page.locator(f'[data-tab="{tab}"]').click()
            page.locator('.point').first.click()
            if name=='mixed':
                page.locator('[data-filter="thread"]').select_option('0')
                assert page.locator('#detail tbody tr').count()==2
                assert all(x=='0' for x in page.locator('#detail tbody tr td:first-child').all_text_contents())
                page.locator('[data-tab="compiler"]').click()
                page.get_by_label('编译版本').select_option('instrumented')
                page.get_by_label('编译阶段').select_option('cuda')
                assert '__global__' in page.locator('#detail pre').inner_text()
            if name=='fragment':
                assert page.locator('.heatmap .missing').count()>0
                page.get_by_label('矩阵数值类型').select_option('abs_error')
                assert page.locator('.heatmap span').count()==64
                page.locator('#launch-filter').select_option('1')
                assert page.locator('.point').count()==1
            if name=='no-reference':
                assert '未提供' in page.locator('#overview').inner_text()
                page.locator('#point-search').fill('not-a-point')
                assert page.locator('.point').count()==0
                page.locator('#point-search').fill('')
            page.screenshot(path=str(output/(name+'.png')),full_page=True)
            page.set_viewport_size(dict(width=600,height=900))
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert not errors,errors
            result.append(dict(case=name,passed=True,points=count,page_errors=errors))
            page.close()
        # Synthetic renderer test only: never pass this off as GPU evidence.
        model=json.loads((root/'fragment/report.json').read_text(encoding='utf-8'))
        model['run']['status']='partial'
        model['source']['text']='</script><script>window.injected=true</script>'
        model['analysis'].update(status='not_provided',result=None)
        point=model['points'][0]
        point['coverage']['capture_integrity']='partial'
        point['values'][0].update(actual='9223372036854775807',expected=None,matched=None)
        fixture=output/'synthetic-renderer-only.html'
        fixture.write_text(html(model),encoding='utf-8')
        page=context.new_page();errors=[]
        page.on('pageerror',lambda e: errors.append(str(e)))
        page.goto(fixture.as_uri());page.wait_for_selector('.point')
        assert page.evaluate('window.injected === undefined')
        assert '9223372036854775807' in page.locator('#detail').inner_text()
        assert '部分' in page.locator('#overview').inner_text()
        assert '未提供' in page.locator('#overview').inner_text()
        assert not errors,errors
        result.append(dict(case='synthetic-renderer-only',passed=True,page_errors=errors))
        page.close()
        browser.close()
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
